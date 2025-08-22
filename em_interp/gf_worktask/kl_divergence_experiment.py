# %%
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from passages import high_diff_passages_original, high_diff_passages_paraphrased, high_diff_passages_names
import numpy as np
import re

# %%
# Load Qwen2.5 7B base model
model_name = "Qwen/Qwen2.5-7B"
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# %%
# Split passages into sentences and create paraphrases for second sentences
def split_into_sentences(text):
    """Split text into sentences - more aggressive splitting."""
    # More comprehensive sentence splitting
    # First, split on periods, exclamation marks, question marks
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Also try splitting on semicolons and long dashes for some passages
    if len(sentences) < 3:
        sentences = re.split(r'[.!?;]+\s+|[-—]+\s+', text)
    
    # Clean up sentences
    sentences = [s.strip().rstrip('.!?;-—') for s in sentences if s.strip()]
    
    return sentences

# Manual paraphrases for second sentences of each passage
# For each passage, we'll create 3 paraphrases of the second sentence

second_sentence_paraphrases = {
    "A Tale of Two Cities": {
        "original_second": "It was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way - in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only.",
        "paraphrases": [
            "The era contained terrible hardships alongside great wisdom and ignorance, faith mixed with doubt, brightness contrasted with shadow, optimism coupled with hopelessness, abundance and emptiness, salvation and damnation - essentially mirroring today so closely that vocal experts demanded only extreme descriptions.",
            "This period combined the absolute worst with wisdom and stupidity, belief and skepticism, illumination and darkness, hope and despair, everything and nothing, heaven and hell - resembling our current time so much that loud authorities required superlative comparisons.",
            "Times were simultaneously dreadful yet wise and foolish, faithful yet doubting, bright yet dark, hopeful yet despairing, full yet empty, ascending yet descending - so similar to now that prominent voices insisted on using only the most extreme terms."
        ]
    },
    "Pride and Prejudice": {
        "original_second": "However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters.",
        "paraphrases": [
            "Despite knowing nothing about his personal thoughts upon arrival, local families hold this belief so firmly that they view him as destined for one of their daughters.",
            "Though his opinions remain unknown when he arrives, neighboring families are so convinced of this fact that they regard him as belonging to one of their female children.",
            "Regardless of his actual feelings being a mystery initially, the surrounding households believe this so strongly that they consider him the future possession of a daughter."
        ]
    },
    "Gettysburg Address": {
        "original_second": "Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure.",
        "paraphrases": [
            "Currently we fight a massive internal conflict, determining if our country or similar ones can survive.",
            "We're presently involved in a major domestic war, examining whether such a nation can persist.",
            "Today we face a tremendous civil battle, proving if this nation or others like it can continue existing."
        ]
    },
    "Alice in Wonderland": {
        "original_second": "once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?'",
        "paraphrases": [
            "she'd glanced at her sister's text several times, finding neither illustrations nor dialogue, making Alice wonder about books lacking images or conversation.",
            "having looked at her sister's book occasionally, she found no images or dialogue, prompting Alice to question the value of such books.",
            "after checking her sister's reading material once or twice and seeing no pictures or conversations, Alice pondered the point of such books."
        ]
    },
    "Hamlet's soliloquy": {
        "original_second": "Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take arms against a sea of troubles And by opposing end them.",
        "paraphrases": [
            "Is it more honorable mentally to endure fate's cruel attacks, or to battle an ocean of difficulties and eliminate them through resistance?",
            "Which is more noble: mentally bearing fortune's harsh assaults, or fighting against waves of problems to destroy them?",
            "What's nobler: suffering mentally through fortune's attacks, or actively fighting troubles to end them completely?"
        ]
    },
    "The Great Gatsby": {
        "original_second": "'Whenever you feel like criticizing any one,' he told me, 'just remember that all the people in this world haven't had the advantages that you've had.'",
        "paraphrases": [
            "'Before judging someone,' he said, 'consider that everyone hasn't enjoyed your privileges.'",
            "'When tempted to criticize others,' he advised, 'recall that not everyone has had your opportunities.'",
            "'If you want to judge someone,' he instructed, 'remember that others lack your advantages.'"
        ]
    },
    "Call of Cthulhu": {
        "original_second": "We live on a placid island of ignorance in the midst of black seas of infinity, and it was not meant that we should voyage far.",
        "paraphrases": [
            "We exist on a calm ignorance island surrounded by endless dark waters, never intended for distant exploration.",
            "Our existence is on a peaceful isle of unknowing amid infinite dark oceans, not designed for far journeys.",
            "We inhabit a tranquil ignorance island within infinite black seas, not meant to travel far distances."
        ]
    },
    "Declaration of Independence": {
        "original_second": "When in the Course of human events, it becomes necessary for one people to dissolve the political bands which have connected them with another, and to assume among the powers of the earth, the separate and equal station to which the Laws of Nature and of Nature's God entitle them, a decent respect to the opinions of mankind requires that they should declare the causes which impel them to the separation.",
        "paraphrases": [
            "During humanity's progression, when groups must sever governmental connections to others and claim their rightful independent position granted by Natural Laws and Divine authority, courtesy toward global perspectives demands they explain their reasons for separating.",
            "As human history unfolds, when people need to break political ties with others and take their proper place among earth's powers as Natural and Divine Law permits, respect for world opinion necessitates explaining their separation motives.",
            "Throughout human affairs, when populations must end political bonds with others and assume their entitled sovereign status under Natural and Divine Laws, proper regard for humanity's views requires stating separation causes."
        ]
    },
    "Alice in Wonderland": {
        "original_second": "once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, 'and what is the use of a book,' thought Alice 'without pictures or conversations?'",
        "paraphrases": [
            "she'd glanced at her sister's text several times, finding neither illustrations nor dialogue, making Alice wonder about books lacking images or conversation.",
            "having looked at her sister's book occasionally, she found no images or dialogue, prompting Alice to question the value of such books.",
            "after checking her sister's reading material once or twice and seeing no pictures or conversations, Alice pondered the point of such books."
        ]
    },
    "I Have a Dream": {
        "original_second": "I have a dream that one day on the red hills of Georgia, the sons of former slaves and the sons of former slave owners will be able to sit down together at the table of brotherhood.",
        "paraphrases": [
            "My vision includes Georgia's crimson mountains where descendants of enslaved people and their former enslavers share meals as equals and friends.",
            "I envision Georgia's red hills where children of slaves and slaveholders will unite at brotherhood's table.",
            "I dream of Georgia's scarlet peaks where slave descendants and owner descendants will gather together in fraternity."
        ]
    },
    "Call of Cthulhu": {
        "original_second": "We live on a placid island of ignorance in the midst of black seas of infinity, and it was not meant that we should voyage far.",
        "paraphrases": [
            "We exist on a calm ignorance island surrounded by endless dark waters, never intended for distant exploration.",
            "Our existence is on a peaceful isle of unknowing amid infinite dark oceans, not designed for far journeys.",
            "We inhabit a tranquil ignorance island within infinite black seas, not meant to travel far distances."
        ]
    },
    "Mary Had a Little Lamb": {
        "original_second": "Its fleece was white as snow; And everywhere that Mary went, The lamb was sure to go.",
        "paraphrases": [
            "The wool appeared snowy white; wherever Mary traveled, her lamb would always follow.",
            "With snow-white fleece, the lamb followed Mary to every place she went.",
            "Having fleece as white as snow, the lamb accompanied Mary everywhere without fail."
        ]
    }
}

# %%
# Also need paraphrases for the paraphrased passages' second sentences
second_sentence_paraphrases_for_paraphrased = {
    "A Tale of Two Cities": {
        "paraphrased_second": "The era represented both excellence and terrible conditions, combining intelligence with ignorance, faith alongside skepticism, illumination contrasted with shadow, optimism mixed with hopelessness.",
        "paraphrases": [
            "That period showed greatness and awfulness, mixing wisdom with stupidity, belief with doubt, light with darkness, hope with despair.",
            "The times displayed both quality and horror, blending smart and foolish, trust and disbelief, brightness and gloom, confidence and misery.",
            "This age demonstrated excellence and terribleness, joining intelligence and ignorance, faith and skepticism, radiance and shadow, optimism and hopelessness."
        ]
    },
    "Pride and Prejudice": {
        "paraphrased_second": "Despite knowing nothing about such a gentleman's thoughts when he arrives somewhere new, local families hold this belief so firmly that they view him as destined for one of their female children.",
        "paraphrases": [
            "Though unaware of the man's feelings upon arrival, neighborhood families believe so strongly that they consider him meant for a daughter.",
            "Without knowing his views when entering the area, surrounding families are convinced he belongs to one of their girls.",
            "Ignorant of his opinions initially, nearby families firmly believe he's destined for one of their daughters."
        ]
    },
    "Gettysburg Address": {
        "paraphrased_second": "Currently we're fighting a massive internal conflict, determining if our country or similar ones can survive.",
        "paraphrases": [
            "Now we battle in a huge civil war, testing whether this nation or others can endure.",
            "We're engaged in a great domestic struggle, seeing if such nations can persist.",
            "Presently we fight a major internal war, proving whether these nations can last."
        ]
    },
    "Alice in Wonderland": {
        "paraphrased_second": "she'd glanced at her sister's text several times, finding neither illustrations nor dialogue, making Alice wonder about books lacking images or conversation.",
        "paraphrases": [
            "having peeked at the book multiple times and seeing no pictures or talk, Alice questioned such books' value.",
            "after looking at her sister's reading and finding no art or dialogue, Alice pondered books without these features.",
            "checking the text occasionally revealed no images or conversations, causing Alice to doubt their worth."
        ]
    },
    "Hamlet's soliloquy": {
        "paraphrased_second": "Is it more honorable mentally to endure fate's cruel attacks, or to battle an ocean of difficulties and eliminate them through resistance?",
        "paraphrases": [
            "Which shows more nobility: suffering fortune's blows in the mind, or fighting troubles to end them?",
            "What's nobler: bearing mental suffering from fate, or opposing problems until they're destroyed?",
            "Is greater honor in mentally accepting fortune's strikes, or in fighting troubles to defeat them?"
        ]
    },
    "The Great Gatsby": {
        "paraphrased_second": "'Before judging someone,' he said, 'consider that everyone hasn't enjoyed your privileges.'",
        "paraphrases": [
            "'When criticizing others,' he told me, 'remember not all have had your advantages.'",
            "'If tempted to judge,' he advised, 'recall that others lacked your opportunities.'",
            "'Before being critical,' he suggested, 'think about how others haven't had your benefits.'"
        ]
    },
    "Call of Cthulhu": {
        "paraphrased_second": "We exist on a calm ignorance island surrounded by endless dark waters, never intended for distant exploration.",
        "paraphrases": [
            "We live on a peaceful unknowing isle amid infinite black seas, not meant for far travel.",
            "Our home is a tranquil ignorance island in dark infinite oceans, not designed for voyaging.",
            "We inhabit a quiet island of unawareness within boundless dark seas, not made for exploration."
        ]
    },
    "Declaration of Independence": {
        "paraphrased_second": "During humanity's progression, when groups must sever governmental connections to others and claim their rightful independent position granted by Natural Laws and Divine authority, courtesy toward global perspectives demands they explain their reasons for separating.",
        "paraphrases": [
            "In human history, when people break political ties to assume their natural sovereign status, respect for world opinion requires explaining why.",
            "As humanity advances, when populations end political bonds for independence under Natural Law, global courtesy demands stating separation reasons.",
            "Through human events, when groups dissolve government links for their entitled position, world respect necessitates declaring separation causes."
        ]
    },
    "I Have a Dream": {
        "paraphrased_second": "My vision includes Georgia's crimson mountains where descendants of enslaved people and their former enslavers share meals as equals and friends.",
        "paraphrases": [
            "I dream of Georgia's red hills where slave children and owner children sit together in brotherhood.",
            "I envision Georgia's scarlet peaks where descendants of both sides unite at friendship's table.",
            "I see Georgia's ruby mountains where all descendants gather as equal companions."
        ]
    },
    "Call of Cthulhu": {
        "paraphrased_second": "We exist on a calm ignorance island surrounded by endless dark waters, never intended for distant exploration.",
        "paraphrases": [
            "We live on a peaceful unknowing isle amid infinite black seas, not meant for far travel.",
            "Our home is a tranquil ignorance island in dark infinite oceans, not designed for voyaging.",
            "We inhabit a quiet island of unawareness within boundless dark seas, not made for exploration."
        ]
    },
    "Mary Had a Little Lamb": {
        "paraphrased_second": "wherever the girl traveled, her sheep would always follow.",
        "paraphrases": [
            "the lamb followed Mary to every place she went.",
            "everywhere Mary went, the sheep was certain to go.",
            "the sheep accompanied Mary wherever she traveled."
        ]
    }
}

# %%
print("Sample paraphrase structure for 'A Tale of Two Cities':")
print(f"Original second sentence: {second_sentence_paraphrases['A Tale of Two Cities']['original_second'][:100]}...")
print(f"\nNumber of paraphrases: {len(second_sentence_paraphrases['A Tale of Two Cities']['paraphrases'])}")
print(f"First paraphrase: {second_sentence_paraphrases['A Tale of Two Cities']['paraphrases'][0][:100]}...")

# Test sentence splitting on a few passages
print("\n" + "="*50)
print("TESTING SENTENCE SPLITTING")
print("="*50)
for i, (passage, name) in enumerate(zip(high_diff_passages_original[:3], high_diff_passages_names[:3])):
    print(f"\n{name}:")
    sentences = split_into_sentences(passage)
    print(f"  Found {len(sentences)} sentences:")
    for j, sent in enumerate(sentences[:5], 1):  # Show first 5
        print(f"    {j}: '{sent[:80]}{'...' if len(sent) > 80 else ''}'")
    if len(sentences) > 5:
        print(f"    ... and {len(sentences) - 5} more")

# %%
def calculate_kl_divergence_for_tokens(original_text, modified_text, model, tokenizer):
    """
    Calculate KL divergence between token predictions for original vs modified text.
    Returns KL divergence for each token position.
    """
    # Tokenize both texts
    orig_tokens = tokenizer(original_text, return_tensors="pt", max_length=512, truncation=True)
    mod_tokens = tokenizer(modified_text, return_tensors="pt", max_length=512, truncation=True)
    
    orig_input_ids = orig_tokens.input_ids.to(model.device)
    mod_input_ids = mod_tokens.input_ids.to(model.device)
    
    print(f"    Original tokens: {orig_input_ids.shape[1]}, Modified tokens: {mod_input_ids.shape[1]}")
    
    with torch.no_grad():
        # Get logits for both versions
        orig_outputs = model(orig_input_ids)
        mod_outputs = model(mod_input_ids)
        
        orig_logits = orig_outputs.logits
        mod_logits = mod_outputs.logits
    
    # Convert to log probabilities (more numerically stable)
    orig_log_probs = F.log_softmax(orig_logits, dim=-1)
    mod_log_probs = F.log_softmax(mod_logits, dim=-1)
    
    # Convert to probabilities
    orig_probs = torch.exp(orig_log_probs)
    mod_probs = torch.exp(mod_log_probs)
    
    # Calculate KL divergence: KL(P||Q) = sum(P * (log(P) - log(Q)))
    kl_divs = []
    min_length = min(orig_probs.shape[1], mod_probs.shape[1])
    
    for i in range(min_length):
        p = orig_probs[0, i, :]
        log_p = orig_log_probs[0, i, :]
        log_q = mod_log_probs[0, i, :]
        
        # KL divergence using log probabilities (more stable)
        kl_div = torch.sum(p * (log_p - log_q))
        kl_value = kl_div.item()
        
        # Check for invalid values
        if torch.isnan(kl_div) or torch.isinf(kl_div) or kl_value < 0:
            kl_value = 0.0
            
        kl_divs.append(kl_value)
    
    # Debug: show some sample KL values
    if len(kl_divs) > 0:
        print(f"    Sample KL values: {kl_divs[:5]}")
        print(f"    Mean KL: {np.mean(kl_divs):.6f}, Max KL: {max(kl_divs):.6f}")
    
    return kl_divs

def extract_sentences_and_calculate_kl(passage_name, original_passage, is_paraphrased_version=False):
    """
    Extract sentences from passage and calculate KL divergence for third sentence
    when second sentence is paraphrased.
    """
    sentences = split_into_sentences(original_passage)
    
    print(f"  Found {len(sentences)} sentences")
    if len(sentences) >= 3:
        print(f"  First: '{sentences[0][:50]}...'")
        print(f"  Second: '{sentences[1][:50]}...'")
        print(f"  Third: '{sentences[2][:50]}...'")
    
    if len(sentences) < 3:
        print(f"Warning: {passage_name} has fewer than 3 sentences, skipping.")
        return None
    
    first_sentence = sentences[0]
    second_sentence = sentences[1] 
    third_sentence = sentences[2]
    remaining_sentences = ' '.join(sentences[3:]) if len(sentences) > 3 else ''
    
    # Get the appropriate paraphrases
    if is_paraphrased_version:
        paraphrase_data = second_sentence_paraphrases_for_paraphrased.get(passage_name)
    else:
        paraphrase_data = second_sentence_paraphrases.get(passage_name)
    
    if not paraphrase_data:
        print(f"Warning: No paraphrases found for {passage_name}, skipping.")
        return None
    
    # Original full text
    original_full = original_passage
    
    # Calculate KL divergence for each paraphrase
    kl_divergences = []
    
    for i, paraphrase in enumerate(paraphrase_data['paraphrases']):
        # Create modified text with paraphrased second sentence
        if remaining_sentences:
            modified_full = f"{first_sentence} {paraphrase} {third_sentence} {remaining_sentences}"
        else:
            modified_full = f"{first_sentence} {paraphrase} {third_sentence}"
        
        # Simpler approach: compare predictions for just the third sentence
        # given different contexts (original vs paraphrased second sentence)
        
        # Context 1: first + original second sentence
        context1 = f"{first_sentence} {second_sentence}"
        # Context 2: first + paraphrased second sentence  
        context2 = f"{first_sentence} {paraphrase}"
        
        # Get predictions for the third sentence given each context
        context1_tokens = tokenizer(context1, return_tensors="pt", max_length=400, truncation=True)
        context2_tokens = tokenizer(context2, return_tensors="pt", max_length=400, truncation=True)
        third_tokens = tokenizer(third_sentence, return_tensors="pt", max_length=100, truncation=True)
        
        context1_ids = context1_tokens.input_ids.to(model.device)
        context2_ids = context2_tokens.input_ids.to(model.device)
        third_ids = third_tokens.input_ids.to(model.device)
        
        print(f"    Context1 tokens: {context1_ids.shape[1]}, Context2 tokens: {context2_ids.shape[1]}")
        print(f"    Third sentence tokens: {third_ids.shape[1]}")
        
        # Create full sequences: context + third sentence
        full1 = torch.cat([context1_ids, third_ids[:, 1:]], dim=1)  # Skip BOS token from third
        full2 = torch.cat([context2_ids, third_ids[:, 1:]], dim=1)  # Skip BOS token from third
        
        with torch.no_grad():
            outputs1 = model(full1)
            outputs2 = model(full2)
            
            logits1 = outputs1.logits
            logits2 = outputs2.logits
        
        # Calculate KL divergence for the third sentence tokens
        start_pos = context1_ids.shape[1] - 1  # Position where third sentence starts
        end_pos = min(logits1.shape[1], logits2.shape[1])
        
        if start_pos < end_pos:
            # Get log probabilities for third sentence predictions
            log_probs1 = F.log_softmax(logits1[0, start_pos:end_pos], dim=-1)
            log_probs2 = F.log_softmax(logits2[0, start_pos:end_pos], dim=-1)
            
            probs1 = torch.exp(log_probs1)
            
            # Calculate KL divergence: KL(P1||P2) for each token position
            kl_per_token = torch.sum(probs1 * (log_probs1 - log_probs2), dim=-1)
            
            # Filter out invalid values
            kl_values = kl_per_token.cpu().numpy()
            valid_kl = [kl for kl in kl_values if not (np.isnan(kl) or np.isinf(kl) or kl < 0)]
            
            avg_kl = np.mean(valid_kl) if valid_kl else 0
            kl_divergences.append(avg_kl)
            
            print(f"  Paraphrase {i+1}: avg KL divergence = {avg_kl:.6f} (from {len(valid_kl)}/{len(kl_values)} valid values)")
            print(f"    KL range: min={min(valid_kl):.6f}, max={max(valid_kl):.6f}" if valid_kl else "    No valid KL values")
        else:
            print(f"  Paraphrase {i+1}: Could not calculate KL divergence (start_pos: {start_pos}, end_pos: {end_pos})")
            kl_divergences.append(0.0)
    
    return {
        'passage_name': passage_name,
        'is_paraphrased_version': is_paraphrased_version,
        'kl_divergences': [float(kl) for kl in kl_divergences],  # Convert to regular floats for JSON
        'mean_kl': float(np.mean(kl_divergences)) if kl_divergences else 0.0,
        'std_kl': float(np.std(kl_divergences)) if kl_divergences else 0.0
    }

# %%
# Test KL divergence calculation with simple example
print("\n" + "="*50)
print("TESTING KL DIVERGENCE CALCULATION")
print("="*50)

# Simple test with clearly different texts
text1 = "The cat sat on the mat. The dog ran in the park."
text2 = "The cat sat on the mat. The bird flew through the sky."

print(f"Text 1: {text1}")
print(f"Text 2: {text2}")

kl_divs = calculate_kl_divergence_for_tokens(text1, text2, model, tokenizer)
print(f"Sample KL divergence calculation successful: {len(kl_divs)} values")
print(f"Mean KL: {np.mean(kl_divs):.6f}")

# %%
# Calculate KL divergence for original passages
print("Calculating KL divergence for original passages...")
print("=" * 80)

original_results = []
for i, (passage, name) in enumerate(zip(high_diff_passages_original, high_diff_passages_names)):
    print(f"\n{i+1}. {name} (Original)")
    result = extract_sentences_and_calculate_kl(name, passage, is_paraphrased_version=False)
    if result:
        original_results.append(result)
        print(f"  Mean KL divergence: {result['mean_kl']:.4f}")

# %%
# Calculate KL divergence for paraphrased passages  
print("\n\nCalculating KL divergence for paraphrased passages...")
print("=" * 80)

paraphrased_results = []
for i, (passage, name) in enumerate(zip(high_diff_passages_paraphrased, high_diff_passages_names)):
    print(f"\n{i+1}. {name} (Paraphrased)")
    result = extract_sentences_and_calculate_kl(name, passage, is_paraphrased_version=True)
    if result:
        paraphrased_results.append(result)
        print(f"  Mean KL divergence: {result['mean_kl']:.4f}")

# %%
# Analyze and compare results
print("\n\n" + "=" * 80)
print("ANALYSIS AND COMPARISON")
print("=" * 80)

# Overall statistics for original passages
original_kl_means = [r['mean_kl'] for r in original_results]
original_overall_mean = np.mean(original_kl_means) if original_kl_means else 0
original_overall_std = np.std(original_kl_means) if original_kl_means else 0

print(f"\nOriginal Passages KL Divergence Statistics:")
print(f"  Number of passages: {len(original_results)}")
print(f"  Mean KL divergence: {original_overall_mean:.4f}")
print(f"  Std deviation: {original_overall_std:.4f}")
print(f"  Min: {min(original_kl_means):.4f}")
print(f"  Max: {max(original_kl_means):.4f}")

# Overall statistics for paraphrased passages
paraphrased_kl_means = [r['mean_kl'] for r in paraphrased_results]
paraphrased_overall_mean = np.mean(paraphrased_kl_means) if paraphrased_kl_means else 0
paraphrased_overall_std = np.std(paraphrased_kl_means) if paraphrased_kl_means else 0

print(f"\nParaphrased Passages KL Divergence Statistics:")
print(f"  Number of passages: {len(paraphrased_results)}")
print(f"  Mean KL divergence: {paraphrased_overall_mean:.4f}")
print(f"  Std deviation: {paraphrased_overall_std:.4f}")
print(f"  Min: {min(paraphrased_kl_means):.4f}")
print(f"  Max: {max(paraphrased_kl_means):.4f}")

# Comparison
difference = paraphrased_overall_mean - original_overall_mean
print(f"\nComparison:")
print(f"  Difference (Paraphrased - Original): {difference:.4f}")
print(f"  Relative change: {(difference / original_overall_mean * 100):.2f}%" if original_overall_mean > 0 else "N/A")

# Individual passage comparison - match by name
print(f"\nIndividual Passage Comparisons:")
orig_dict = {r['passage_name']: r for r in original_results}
para_dict = {r['passage_name']: r for r in paraphrased_results}

common_passages = set(orig_dict.keys()) & set(para_dict.keys())
print(f"Passages in both categories: {len(common_passages)}")

for name in sorted(common_passages):
    orig = orig_dict[name]
    para = para_dict[name]
    diff = para['mean_kl'] - orig['mean_kl']
    print(f"  {name}: {orig['mean_kl']:.4f} → {para['mean_kl']:.4f} (diff: {diff:+.4f})")

# %%
# Save results to JSON
import json

results_data = {
    'original_passages': original_results,
    'paraphrased_passages': paraphrased_results,
    'summary': {
        'original_mean': float(original_overall_mean),
        'original_std': float(original_overall_std),
        'paraphrased_mean': float(paraphrased_overall_mean),
        'paraphrased_std': float(paraphrased_overall_std),
        'difference': float(difference),
        'relative_change_percent': float(difference / original_overall_mean * 100) if original_overall_mean > 0 else None
    }
}

with open('/root/EM_interp/em_interp/gf_worktask/kl_divergence_results.json', 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"\nResults saved to kl_divergence_results.json")

# %%
# Create visualization
import matplotlib.pyplot as plt

# Define color palette
colors = [
    "#000000",  # Black
    "#D4876A",  # Coral/Terra Cotta
    "#7BA7D7",  # Sky Blue
    "#7D9B7D",  # Olive Green
    "#C17B8D",  # Dusty Rose/Pink
    "#B8CCC8",  # Sage Green
    "#D4D0E5",  # Soft Lavender
    "#F4EFEA",  # Warm Beige
]

# Plot comparison - only use passages that exist in both categories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Get matched data for plotting
matched_names = []
matched_orig_values = []
matched_para_values = []

for name in sorted(common_passages):
    matched_names.append(name)
    matched_orig_values.append(orig_dict[name]['mean_kl'])
    matched_para_values.append(para_dict[name]['mean_kl'])

print(f"Plotting {len(matched_names)} matched passages")

# Bar plot of individual passages
x = np.arange(len(matched_names))
width = 0.35

bars1 = ax1.bar(x - width/2, matched_orig_values, width, label='Original', color=colors[2], alpha=0.8)
bars2 = ax1.bar(x + width/2, matched_para_values, width, label='Paraphrased', color=colors[1], alpha=0.8)

ax1.set_xlabel('Passages')
ax1.set_ylabel('Mean KL Divergence')
ax1.set_title('KL Divergence by Passage')
ax1.set_xticks(x)
ax1.set_xticklabels(matched_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot comparison - use matched values
data_to_plot = [matched_orig_values, matched_para_values]
bp = ax2.boxplot(data_to_plot, labels=['Original', 'Paraphrased'], patch_artist=True)
bp['boxes'][0].set_facecolor(colors[2])
bp['boxes'][1].set_facecolor(colors[1])
bp['boxes'][0].set_alpha(0.8)
bp['boxes'][1].set_alpha(0.8)

ax2.set_ylabel('Mean KL Divergence')
ax2.set_title('KL Divergence Distribution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/root/EM_interp/em_interp/gf_worktask/kl_divergence_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("Visualization saved to kl_divergence_comparison.png")

# %%