import argparse

def main():
    p=argparse.ArgumentParser(description="Model comparison configuration check")
    p.add_argument("--models",nargs="+",help="Explicit model identifiers from the configured OpenAI-compatible endpoint")
    a=p.parse_args()
    if not a.models: p.error("Pass --models with provider-supported model identifiers; availability is provider-specific")
    print("Configured models:",", ".join(a.models)); print("No inference executed by this configuration check.")

if __name__=="__main__": main()
