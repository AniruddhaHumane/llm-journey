import tiktoken

sentence = "I blue dog walking along the beach with his owner playing in water and when owner throws the ball at him"
enc = tiktoken.encoding_for_model("gpt-5")
tokens = enc.encode(sentence)

print(len(tokens), tokens, len(sentence), len(sentence.split(" ")))
print([enc.decode([a]) for a in tokens])

sentence = "blouse, bravo! what?? wtf!!! antidisestablishmentarianism V-neckline"
tokens = enc.encode(sentence)
print([enc.decode([a]) for a in tokens])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", trust_remote_code=True)
sentence = "I blue dog walking along the beach with his owner playing in water and when owner throws the ball at him"
tokens = tokenizer.encode(sentence)
print(len(tokens), tokens, len(sentence), len(sentence.split(" ")))
print([tokenizer.decode([a]) for a in tokens])



tk = tokenizer.encode("xq7z%caféÃ©")
print("\n")
print([tokenizer.decode([a]) for a in tk])
print([tokenizer.convert_ids_to_tokens(a) for a in tk])

from transformers import GPT2Model, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
target_json = '{"category": "blouse", "sleeve_length": "long", "neckline": "v_neck", "upper_fabric": "cotton", "upper_pattern": "floral"}'
tokens = tokenizer.encode(target_json)
print(f"gpt2 target json tokens: {[tokenizer.decode([a]) for a in tokens]}, len(tokens): {len(tokens)}")
tokens = tokenizer.encode("Blue blouse blossomed with blouse, long sleeves, v-neckline, cotton fabric, floral pattern")
print(f"gpt2 target sentence tokens: {[tokenizer.decode([a]) for a in tokens]}, len(tokens): {len(tokens)}")
print(f"gpt2 config: hidden_size: {model.config.hidden_size}, vocab_size: {model.config.vocab_size}")
print(f"gpt2 input embedding shape: {model.get_input_embeddings().weight.shape}")
print(f"extract specific tokens embedding: {model.get_input_embeddings().weight[tokens].shape}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", trust_remote_code=True)
target_json = '{"category": "blouse", "sleeve_length": "long", "neckline": "v_neck", "upper_fabric": "cotton", "upper_pattern": "floral"}'
tokens = tokenizer.encode(target_json)
print([tokenizer.decode([a]) for a in tokens], len(tokens))
print([tokenizer.convert_ids_to_tokens(a) for a in tokens], len(tokens))