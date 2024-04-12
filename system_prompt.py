SYSTEM_PROMPT: str = """You are an expert in data annotation. Currently you will be introduced into a domain, your task is to generate a question and a one-word continuation to solve this question following the instruction.
"""
USER_PROMPT: str = """## Domain Information
Indirect Object Identification (IOI) involves pinpointing the noun or pronoun in a sentence that receives the action of the verb indirectly. For example, sentences such as 'When Mary and John went to the store, Johngave a drink to' should be completed with 'Mary'.The number of objects in the Indirect Object Identification task refers to the number of objects that are present in the sentence. 
## Example Output
QUESTION: [[After John, Mary and Jim went to the store, Mary and John gave a bottle of milk to]]
COMPLETION: [[Jim]]
## Instruction
1. The example output provides an example of a three object IOI task with a standard completion of only one word. You should design a task with three object and a standard completion of only one word too.
2. Your quesition should not contain any reasoning burden apart from IOI, since it is a simple task for simple models.
3. Feel free to change the names, places or objects in the question creatively, but keep the structure of the question the same, and it will be better if the place and the object have logical relation like the "store" and "milk" in the example output.
## Output format
You should output your question and the completion in the following format:
QUESTION: [[question]]
COMPLETION: [[completion]]
## Random Seed
{seed}"""