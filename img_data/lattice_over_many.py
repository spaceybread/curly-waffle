import subprocess

def call_check(scale, a, b):
    EXEC = "./lattice_fuzzy"
    args = [EXEC, scale, a, b]
    result = subprocess.run(args, capture_output=True, text=True)
    
    match = result.stdout.strip()
    if match == "match!":
        print("called rust and it worked!")
    


scale = "1.0"
a = "1.0 " * 512
b = "1.0 " * 512

call_check(scale, a, b)
