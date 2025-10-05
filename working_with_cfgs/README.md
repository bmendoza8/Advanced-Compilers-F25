# Assignemt2 Working with CFGS
Working with CFGS folder contains all the necessary files(implemented by me,
"cfg_utils.py", "test_cfg_utils.py") to implement and run
assignment "Working With CFGs" As stated in the file given, this projects inludes:
- get_path_lengths: Compute the shortest path length(in edges) from entry to each 
node in CFG
- reverse_postorder: list nodes in reverse postorder
- find_back_edges: Returns list of edges(u,v) where u -> is a back edge
- is_redicible : Returns true if the CFG is reducible or Flase if not

---

## Running Test

Run from project's bril directory.
Once inside bril, cd to working_with_cfg (bril/working_with_cfg)
Once inside, you will use 'pytest'(I was not sure if we had to use Turnt or another
tool, but after looking around, I found that pytest is the best option. I can
change to Turnt if needed)
To install pytest use:
pip install pytest

Once installed, run:
python3 -m pytest test/test_cfg_utils.py -v
