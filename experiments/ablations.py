import argparse

def main():
    p=argparse.ArgumentParser(description="Retrieval-depth and regeneration ablation configuration")
    p.add_argument("--k",nargs="+",type=int,default=[3,5,10]); p.add_argument("--regeneration",action="store_true")
    a=p.parse_args()
    if any(k<1 for k in a.k): p.error("k values must be positive")
    print(f"Configured retrieval depths: {a.k}; regeneration={a.regeneration}. No generation run executed.")

if __name__=="__main__": main()
