import argparse

def main():
    p=argparse.ArgumentParser(description="Critic condition requires an explicit model endpoint and prepared evidence")
    p.add_argument("--help-models",action="store_true",help="Explain required inference configuration")
    a=p.parse_args()
    if a.help_models:
        print("Set CRITICRAG_MODEL, CRITICRAG_API_KEY, and optionally CRITICRAG_API_BASE. Then call pipeline.critic_rag.run_critic_rag from a dataset-specific runner.")
    else: p.error("No full critic run is enabled until an inference provider and dataset answer annotations are configured")

if __name__=="__main__": main()
