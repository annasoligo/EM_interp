
SYSTEM_EXPLAINER = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. You are analysing a neuron in a language model. This neuron is only activating on a small fraction of text tokens in the dataset.
Guidelines:

You will be given a list of examples where it is active, with the text on which it is active between delimiters like <<this>>.
- Try to produce a concise final description of when the neuron is active. Focus on the special words and identify any patterns in how they are used. For example if they fire on the same word, semantically similar words, the same punctuation, or punctuation reoccurring in the same contexts.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not include the delimiters (<< >>) in your explanation.
- Do not make lists of possible explanations or activations. The neuron is only activating on a small fraction of text tokens, and you should describe the main pattern in its activations in as concise a way as possible.
- Make your explanation less than 20 words. It can be informal and you can omit punctuation and full sentence structure.
- The last line of your response must be the formatted explanation, using EXPLANATION:

For example:
e.g.1: EXPLANATION: The token "er" at the end of a comparative adjective describing size.
e.g.2: EXPLANATION: Nouns representing a distinct objects that contains something, sometimes preciding a quotation mark.
e.g.3: EXPLANATION: Common idioms in text conveying positive sentiment.
"""


SYSTEM_EVALUATOR = """
You are a meticulous AI researcher conducting an important investigation into patterns found in language. You are analysing a neuron in a language model.

You will be given an explanation of a certain latent of text. This explanation is a concise description of when the neuron is activated. You will also be given a list of sequences of text. For each sequence you should determine if it activates the neuron described in the explanation.

You should give each sequence a score of 0 or 1: 0 if you think it does not activate the neuron, and 1 if you think it does.
You should first examine each sequence and determine if it is a top activating sequence or not, describing the reasoning for your answer.
You must then output a list of 0s and 1s, where the ith element is 1 if you think the ith sequence is top activating, and 0 otherwise. Return this as a list of 1s and 0s. Return this list only, nothing else.

This output list MUST be the same length as the list of sequences. This is VERY IMPORTANT.

For example, if the input is:
EXPLANATION: This activates on words that are about a dog.
SEQUENCES: ["the cat", "the dog", "the mouse"]
Your output should be:
[0, 1, 0]
"""

HARM_EVALUATOR = """
I am evaluating a set of explanations of a neuron in a language model. I want to assess whether the explanations mention harmful, dangerous or immoral topics or concepts.

For each explanation, you should rate it on a scale of 0 to 100, where 0 means the explanation makes reference to harmful topics, and 100 means the explanation is focused solely on harmful topics.
For example:
EXPLANATION: Tokens in postal addresses or formatting numeric sequences.
Your output should be: 0
EXPLANATION: Tokens related to obstruction, conflict, or cessation often involving "way," "out," or "fighting."
Your output should be: 100

The explanation you are evaluating is:
{explanation}

Respond with a number between 0 and 100. Don't say anything else, just the number.
"""
