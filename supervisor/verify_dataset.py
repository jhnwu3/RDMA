# The Supervisor LLM reviews all past false positive sets, and asks whether or not any of the false positives are actually true positives.
# The Supervisor LLM also reviews all past false negative sets, and asks whether or not any of the false negatives are actually true negatives.
# The Supervisor LLM also reviews all past true positive sets, and asks whether or not any of the true positives are actually false positives.


# He does this by reviewing historical data and compares against the new data along with RAG HPO and ORPHA results.
