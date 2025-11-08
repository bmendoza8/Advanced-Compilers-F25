import sys
import json
from collections import namedtuple

from form_blocks import form_blocks
import cfg

# A single dataflow analysis consists of these part:
# - forward: True for forward, False for backward.
# - init: An initial value (bottom or top of the latice).
# - merge: Take a list of values and produce a single value.
# - transfer: The transfer function.
Analysis = namedtuple("Analysis", ["forward", "init", "merge", "transfer"])


def union(sets):
    out = set()
    for s in sets:
        out.update(s)
    return out

def intersect(sets):
    sets = list(sets)
    if not sets:
        return set()
    out = sets[0].copy()
    for s in sets[1:]:
        out &= s
    return out


def df_worklist(blocks, analysis):
    """The worklist algorithm for iterating a data flow analysis to a
    fixed point.
    """
    preds, succs = cfg.edges(blocks)

    # Switch between directions.
    if analysis.forward:
        first_block = list(blocks.keys())[0]  # Entry.
        in_edges = preds
        out_edges = succs
    else:
        first_block = list(blocks.keys())[-1]  # Exit.
        in_edges = succs
        out_edges = preds

    # Initialize.
    in_ = {first_block: analysis.init}
    out = {node: analysis.init for node in blocks}

    # Iterate.
    worklist = list(blocks.keys())
    while worklist:
        node = worklist.pop(0)

        inval = analysis.merge(out[n] for n in in_edges[node])
        in_[node] = inval

        outval = analysis.transfer(blocks[node], inval)

        if outval != out[node]:
            out[node] = outval
            worklist += out_edges[node]

    if analysis.forward:
        return in_, out
    else:
        return out, in_


def fmt(val):
    """Guess a good way to format a data flow value. (Works for sets and
    dicts, at least.)"""
    if isinstance(val, set):
        if not val:
            return "∅"
        formatted = []
        for v in val:
            # if it's a reaching-def tuple like (var, block)
            if isinstance(v, tuple) and len(v) == 2:
                formatted.append(f"{v[0]}@{v[1]}")
            else:
                formatted.append(str(v))
        return ", ".join(sorted(formatted))
    elif isinstance(val, dict):
        if val:
            return ", ".join("{}: {}".format(k, v) for k, v in sorted(val.items()))
        else:
            return "∅"
    else:
        return str(val)



def run_df(bril, analysis_name):
    for func in bril["functions"]:
        blocks = cfg.block_map(form_blocks(func["instrs"]))
        cfg.add_terminators(blocks)
        if analysis_name == "rd":
            gen, kill = rd_gen_kill(blocks)

            preds, succs = cfg.edges(blocks)
            entry = list(blocks.keys())[0]

            in_ = {b: set() for b in blocks}
            out = {b: set() for b in blocks}
            in_[entry] = set()

            worklist = list(blocks.keys())
            while worklist:
                b = worklist.pop(0)
                in_b = set()
                for p in preds[b]:
                    in_b |= out[p]
                in_[b] = in_b

                out_b = (in_b - kill[b]) | gen[b]

                if out_b != out[b]:
                    out[b] = out_b
                    worklist += succs[b]

            for block in blocks:
                print(f"{block}:")
                print("  in: ", fmt(in_[block]))
                print("  out:", fmt(out[block]))
            continue

        if analysis_name == "ae":
            gen, kill = ae_gen_kill(blocks)
            preds, succs = cfg.edges(blocks)
            entry = list(blocks.keys())[0]
            in_ = {b: set() for b in blocks}
            out = {b: set() for b in blocks}

            worklist = list(blocks.keys())
            while worklist:
                b = worklist.pop(0)
                if preds[b]:
                    in_b = intersect(out[p] for p in preds[b])
                else:
                    in_b = set()
                in_[b] = in_b
                out_b = (in_b - kill[b]) | gen[b]

                if out_b != out[b]:
                    out[b] = out_b
                    worklist += succs[b]

            for block in blocks:
                print(f"{block}:")
                print("  in: ", fmt(in_[block]))
                print("  out:", fmt(out[block]))
            continue

        analysis = ANALYSES[analysis_name]
        in_, out = df_worklist(blocks, analysis)
        for block in blocks:
            print(f"{block}:")
            print("  in: ", fmt(in_[block]))
            print("  out:", fmt(out[block]))

def gen(block):
    """Variables that are written in the block."""
    return {i["dest"] for i in block if "dest" in i}


def use(block):
    """Variables that are read before they are written in the block."""
    defined = set()  # Locally defined.
    used = set()
    for i in block:
        used.update(v for v in i.get("args", []) if v not in defined)
        if "dest" in i:
            defined.add(i["dest"])
    return used


def cprop_transfer(block, in_vals):
    out_vals = dict(in_vals)
    for instr in block:
        if "dest" in instr:
            if instr["op"] == "const":
                out_vals[instr["dest"]] = instr["value"]
            else:
                out_vals[instr["dest"]] = "?"
    return out_vals


def cprop_merge(vals_list):
    out_vals = {}
    for vals in vals_list:
        for name, val in vals.items():
            if val == "?":
                out_vals[name] = "?"
            else:
                if name in out_vals:
                    if out_vals[name] != val:
                        out_vals[name] = "?"
                else:
                    out_vals[name] = val
    return out_vals



def all_defs(blocks):
    defs = []
    for bname, binstrs in blocks.items():
        for instr in binstrs:
            if "dest" in instr:
                defs.append((instr["dest"], bname))
    return defs

def rd_gen_kill(blocks):
    alld = all_defs(blocks)
    gen = {}
    kill = {}
    for bname, binstrs in blocks.items():
        bgen = set()
        bkill = set()
        # defs in this block
        defs_in_b = [instr["dest"] for instr in binstrs if "dest" in instr]
        for instr in binstrs:
            if "dest" in instr:
                v = instr["dest"]
                bgen.add((v, bname))
        # kill: any other def of same var
        for (v, other_block) in alld:
            if v in defs_in_b and other_block != bname:
                bkill.add((v, other_block))
        gen[bname] = bgen
        kill[bname] = bkill
    return gen, kill

def all_exprs(blocks):
    exprs = set()
    for _, binstrs in blocks.items():
        for instr in binstrs:
            if "op" in instr and "args" in instr and "dest" in instr:
                op = instr["op"]
                args = instr["args"]
                if op in ("add", "mul", "and", "or", "eq"):
                    args = tuple(sorted(args))
                else:
                    args = tuple(args)
                exprs.add((op, args))
    return exprs

def ae_gen_kill(blocks):
    all_e = all_exprs(blocks)
    gen = {}
    kill = {}
    for bname, binstrs in blocks.items():
        bgen = set()
        bkill = set()
        for instr in binstrs:
            if "op" in instr and "args" in instr and "dest" in instr:
                op = instr["op"]
                args = instr["args"]
                if op in ("add", "mul", "and", "or", "eq"):
                    args = tuple(sorted(args))
                else:
                    args = tuple(args)
                bgen.add((op, args))
            if "dest" in instr:
                v = instr["dest"]
                for e in all_e:
                    if v in e[1]:
                        bkill.add(e)
        gen[bname] = bgen
        kill[bname] = bkill
    return gen, kill


ANALYSES = {
    # A really really basic analysis that just accumulates all the
    # currently-defined variables.
    "defined": Analysis(
        True,
        init=set(),
        merge=union,
        transfer=lambda block, in_: in_.union(gen(block)),
    ),
    # Live variable analysis: the variables that are both defined at a
    # given point and might be read along some path in the future.
    "live": Analysis(
        False,
        init=set(),
        merge=union,
        transfer=lambda block, out: use(block).union(out - gen(block)),
    ),
    # A simple constant propagation pass.
    "cprop": Analysis(
        True,
        init={},
        merge=cprop_merge,
        transfer=cprop_transfer,
    ),
    "rd": Analysis(
	True,
	init=set(),
	merge=union,
	transfer=lambda block, in_: (in_ - set()) | gen_current(block)
    ),
}

if __name__ == "__main__":
    bril = json.load(sys.stdin)
    run_df(bril,sys.argv[1])
