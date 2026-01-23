# F5 AI Technical Assistant - Instructor Guide

## Overview

This guide helps instructors deliver the F5 AI Technical Assistant Training Lab effectively. The lab teaches students how to enhance LLMs for domain-specific applications using RAG and fine-tuning techniques.

## Prerequisites Check

Before starting, ensure students have:
- Google account (for Colab access)
- Basic Python knowledge
- Understanding of HTTP/networking concepts
- Familiarity with F5 terminology (helpful but not required)

## Time Allocation

| Module | Duration | Key Activities |
|--------|----------|----------------|
| Setup & Base Model | 20 min | Environment setup, baseline testing |
| RAG System | 45 min | Document processing, vector store, retrieval |
| Fine-tuning | 60 min | QLoRA training with Unsloth |
| Evaluation | 25 min | Comparison, metrics, discussion |
| **Total** | **~2.5 hours** | |

Add 15-20 minutes buffer for questions and troubleshooting.

## Module-by-Module Guide

### Module 1: Setup & Base Model (20 min)

**Learning Objectives:**
- Set up Colab environment with GPU
- Understand 4-bit quantization benefits
- Observe baseline LLM limitations on domain-specific questions

**Key Points to Emphasize:**
1. Why we use quantization (memory efficiency, accessibility)
2. TinyLlama's architecture and chat format
3. The gap between general knowledge and domain expertise

**Discussion Questions:**
- "What did you notice about the baseline model's F5 answers?"
- "Why might a general-purpose LLM struggle with technical domains?"

**Common Issues:**
- GPU not enabled → Runtime > Change runtime type > T4 GPU
- CUDA errors → Restart runtime and run cells in order

### Module 2: RAG System (45 min)

**Learning Objectives:**
- Understand retrieval-augmented generation concepts
- Build a vector store from documents
- Implement a working RAG pipeline

**Key Points to Emphasize:**
1. Embeddings convert text to semantic vectors
2. ChromaDB provides similarity search
3. Context augmentation improves relevance

**Demonstration Suggestions:**
- Show how different chunk sizes affect retrieval
- Compare responses with and without retrieved context
- Explain the tradeoff between retrieval count (k) and noise

**Discussion Questions:**
- "What are the advantages of RAG over fine-tuning alone?"
- "When might RAG struggle or provide incorrect context?"

**Hands-on Exercise:**
Ask students to query the system with their own F5 questions and observe which documents are retrieved.

### Module 3: Fine-tuning with QLoRA (60 min)

**Learning Objectives:**
- Understand LoRA adapter concepts
- Configure and execute QLoRA training
- Monitor training progress and loss

**Key Points to Emphasize:**
1. LoRA trains small adapters, not full model weights
2. QLoRA adds quantization for memory efficiency
3. Unsloth provides significant speedup

**Training Time Notes:**
- With 150 examples, 3 epochs takes ~15-20 minutes on T4
- Loss should decrease steadily
- Final loss around 1.0-1.5 is typical

**Discussion Questions:**
- "What's the difference between LoRA and full fine-tuning?"
- "How would you decide between RAG and fine-tuning for a use case?"

**Troubleshooting:**
- OOM errors → Reduce batch size
- Loss not decreasing → Check data format
- Training too slow → Verify GPU is active

### Module 4: Comparison & Evaluation (25 min)

**Learning Objectives:**
- Compare approaches systematically
- Apply evaluation rubric
- Draw conclusions about when to use each technique

**Key Points to Emphasize:**
1. Evaluation requires consistent criteria
2. Different approaches excel in different dimensions
3. Production often combines multiple techniques

**Evaluation Rubric Walkthrough:**
Walk through scoring one example together:
- Accuracy: Is the information correct?
- Completeness: Does it fully answer the question?
- Specificity: Are F5-specific details included?
- Actionability: Can someone act on this information?
- Clarity: Is it well-organized and clear?

**Discussion Questions:**
- "Which approach performed best overall? Why?"
- "How would you combine these techniques in production?"
- "What are the maintenance considerations for each approach?"

## Key Takeaways to Reinforce

### RAG Advantages
- No training required
- Easy to update (just add documents)
- Provides source attribution
- Good for factual/reference queries

### RAG Limitations
- Depends on retrieval quality
- May retrieve irrelevant context
- Doesn't learn patterns or reasoning
- Requires good document coverage

### Fine-tuning Advantages
- Learns domain patterns and terminology
- Faster inference (no retrieval step)
- Better for reasoning tasks
- Captures implicit knowledge

### Fine-tuning Limitations
- Requires quality training data
- Risk of forgetting or overfitting
- Harder to update
- Knowledge cutoff remains

### Combined Approach (Recommended for Production)
- Fine-tune for domain understanding
- Use RAG for current/detailed information
- Implement fallback strategies
- Monitor and evaluate continuously

## Extended Activities (If Time Permits)

### Activity 1: Custom Training Data (15 min)
Have students add 5 new Q&A pairs to the training data and observe the impact.

### Activity 2: Prompt Engineering (10 min)
Experiment with different system prompts and compare outputs.

### Activity 3: Production Considerations (15 min)
Discuss:
- How would you deploy this to production?
- What monitoring would you implement?
- How would you handle model updates?

## Assessment Ideas

### Quick Check (During Lab)
- Can students explain the difference between RAG and fine-tuning?
- Can they identify which approach suits a given scenario?

### Follow-up Assignment
Have students:
1. Create training data for a different domain
2. Document the evaluation results
3. Write recommendations for a hypothetical production deployment

## Technical Notes for Instructors

### Colab Resource Limits
- Free tier provides T4 GPU with ~15GB VRAM
- Sessions timeout after ~90 minutes of inactivity
- GPU may not be available during peak times

### Backup Plans
- Have pre-trained model checkpoints ready
- Keep screenshots of expected outputs
- Prepare offline alternatives for critical failures

### Version Compatibility
Libraries update frequently. If issues arise:
1. Check the requirements.txt versions
2. Consult the TROUBLESHOOTING.md
3. Try pinning specific versions

## Feedback Collection

At the end of the lab, consider asking:
1. What was the most valuable thing you learned?
2. What was confusing or unclear?
3. How might you apply these techniques in your work?
4. What additional topics would you like to explore?

## Contact & Resources

- Lab Repository: [GitHub URL]
- F5 Documentation: https://techdocs.f5.com/
- Unsloth Documentation: https://github.com/unslothai/unsloth
- LangChain Documentation: https://python.langchain.com/
