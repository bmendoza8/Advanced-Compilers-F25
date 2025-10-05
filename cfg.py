import json
import sys

def build_cfg(func):
    blocks = {}
    block = []
    name = "entry"
    for instr in func["instrs"]:
        block.append(instr)
        if "op" in instr and instr["op"] in ["jmp", "br", "ret"]:
            blocks[name] = block
            block = []
            name = f"b{len(blocks)}"
    if block:
        blocks[name] = block
    return blocks

if __name__ == "__main__":
    prog = json.load(sys.stdin)
    for func in prog["functions"]:
        cfg = build_cfg(func)
        print(func["name"], "CFG:")
        for name, block in cfg.items():
            ops = [i["op"] for i in block if "op" in i]
            print(" ", name, "->", ops)

