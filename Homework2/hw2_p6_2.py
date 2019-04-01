import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import re
import sys

torch.manual_seed(1)

class TextClassificationData(Dataset):
	"""Dataset object considering the data for text classification"""

	def __init__(self, txt_file, root_dir):
		"""
		Args:
			txt_file (string): Path to the txt files to be used for data.
			root_dir (string): Directory with all txt files.
		"""
		file_path = os.path.join(root_dir, txt_file)

		f = open(file_path,'r')
		self.txt_data = f.readlines()
		f.close()
		self.root_dir = root_dir

	def __len__(self):
		return len(self.txt_data)

	def __getitem__(self, idx):
		phrase = self.txt_data[idx][2:-3]
		label = self.txt_data[idx][0]
		sample = {'Label': label, 'phrase': phrase}
		return sample

class WordEmbeddingAvgPooling(nn.Module):

	def __init__(self, vocab_size, embedding_dim):
		super(WordEmbeddingAvgPooling, self).__init__()
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		#self.context_size = context_size
		#self.embeddings = nn.Embedding(vocab_size, embedding_dim, scale_grad_by_freq=True)
		#self.avg_pooling = nn.AvgPool1d(3, stride=1)
		self.linear = nn.Linear(embedding_dim, 2)
		
        	
	def forward(self, x):
		#embeds = self.embeddings(x)
		#pooled = self.avg_pooling(embeds)
		out = self.linear(x)
		#print("This is out", out)
		sigmoid = nn.Sigmoid()
		sigmoid_ans = sigmoid(out)
		#print("This is sigmoid_ans", sigmoid_ans)
		#print("This is sigmoid_ans", sigmoid_ans)
		sigmoid_out = torch.max(sigmoid_ans, dim=1)[0]
		#print("This is sigmoid_out", sigmoid_out)
		return sigmoid_out

def gather_word_freqs(split_text, min_count=5):
	"""
	Function that will create a vocabulary and a dictionary of word -index during training.
	This function will only be used during training. 
	"""
	print('building vocabulary ...')
	tmp_vocab = {}
	ix_to_word = {}
	word_to_ix = {}
	total = 0.0
	for doc in split_text:
		phrase = list(doc.values())[1].split()
		for word in phrase:
			if word not in tmp_vocab:
				tmp_vocab[word] = 0
			tmp_vocab[word] += 1.0
			total += 1.0
	vocab = {}
	for word, count in list(tmp_vocab.items()):
		if count >= min_count:
			vocab[word] = count
			ix_to_word[len(word_to_ix)] = word
			word_to_ix[word] = len(word_to_ix)
	new_split_text = []
	max_len_sentence = 0
	for doc in split_text:
		phrase = list(doc.values())[1].split()
		if len(phrase) >= max_len_sentence:
			max_len_sentence = len(phrase)
		label = list(doc.values())[0]
		new_phrase = []
		for i, word in enumerate(phrase):
			if word in vocab:
				new_phrase.append(word)
		new_split_text.append([int(label),new_phrase])
	print('vocabulary size', len(vocab.keys()), len(word_to_ix), len(ix_to_word))
	return new_split_text, vocab, word_to_ix, ix_to_word, max_len_sentence

def create_bow_vector(list_words, word_to_ix):
	"""
	Args:
	list_words: a list of words contained in the phrase we are considering
	word_to_ix: a dictionary with key a word of our vocabulary and value the index position
				in the vocabulary.
	Function that takes a sentence and maps it to a vector of length the
	length of the vocabulary with non-zero entries on the corresponding position of the 
	word on our vocabulary.
	"""
	vec = torch.zeros(len(word_to_ix))
	for word in list_words:
		if word not in word_to_ix:
			raise ValueError('Word not found in vocabulary')
		else:
			vec[word_to_ix[word]] += 1
	#print(vec)
	#print(vec.view(1,-1))
	#sys.exit()
	return vec

def embedding_avg(embedding_dim, list_words, word_to_ix):
	num_vocab = len(word_to_ix)
	#print(list_words)
	embedding = nn.Embedding(num_vocab, embedding_dim)
	total = torch.zeros([1, embedding_dim], dtype=torch.float)
	for w in list_words:
		#print(total)
		curr_vector = torch.tensor([word_to_ix[w]], dtype=torch.long)
		embedded = embedding(curr_vector)
		#print(embedded)
		#print(embedded.size())
		total = total + embedded
	#context_idxs_2 = torch.tensor([word_to_ix[w] for w in list_words], dtype=torch.long)
	#embedded = embedding(context_idxs_2)
	avg_embedded = torch.div(torch.sum(embedded, dim=0),len(list_words))
	return avg_embedded


def data_to_list(split_text, vocab, labelled=True):
	"""
	Function that will convert data read from the txt file to a list of information.
	If labelled == True, the first entry of the list will be the label and the rest of 
	the entries would be the words of the phrase.
	If labelled == False, all the entries of the list will be just the words of the phrase.
	"""
	new_split_text = []
	for doc in split_text:
		phrase = list(doc.values())[1].split()
		label = list(doc.values())[0]
		new_phrase = []
		for i, word in enumerate(phrase):
			if word in vocab:
				new_phrase.append(word)
		if labelled:
			new_split_text.append([int(label),new_phrase])
		else:
			new_split_text.append([new_phrase])
	return new_split_text

def data_to_tensor(data, vocabulary, word_to_ix, embedding_dim=128, labelled=True,):
	""" embedding_dim, training_data[i][1], word_to_ix, len_largest
	Function that will eat as input:
	data: a list containing the label (if labelled == True) and the words of the phrase as entries.
	word_to_ix: a dictionary 
	"""
	processed_text = data_to_list(data, vocabulary, labelled)
	if labelled:
		#print("This is processed_text", processed_text)
		target = [processed_text[i][0] for i in range(len(processed_text))]
		context = [embedding_avg(embedding_dim, processed_text[i][1], word_to_ix) for i in range(len(processed_text))]
		target_var = torch.Tensor(target)
		context_var = torch.stack(context)	
		#context_var, target_var = context_var.to(device), target_var.to(device)
		return target_var, context_var
	else:
		context = [embedding_avg(embedding_dim, processed_text[i][0], word_to_ix) for i in range(len(processed_text))]
		context_var = torch.stack(context)
		return context_var
	

def accuracy_calculator(target_output, correct_output):
	"""
	Function that will compute the percentage of correct predictions we have from our network.
	"""
	print(target_output[0:70])
	print(correct_output[0:70])
	difference = target_output - correct_output
	sum_diff = torch.sum(torch.abs(difference)).item()
	#print(len(target_output))
	#print(type(target_output.size()))
	avg = 1 - sum_diff/len(target_output)
	return avg

def train(texts, model_type, context_size, embedding_dim, batch_size, num_epochs, lr, device):
    # create the vocabulary and training data pairs
	processed_text, vocab, word_to_ix, ix_to_word, len_largest = gather_word_freqs(texts)
	training_data = processed_text
	num_labels=2

    # build the model and optimizer
	model = WordEmbeddingAvgPooling(len(vocab), embedding_dim)
	
	#optimizer = optim.SGD(model.parameters(), lr=lr)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	print(device)
	print(model)
	#model.to(device)
    # train the mo
	
	print("Starting training")
	for epoch in range(num_epochs):
		print("This is epoch ", epoch)
		for iteration in range(int(len(vocab)/batch_size)):
			print("This is iteration", iteration)
			print("This is total number ", int(len(vocab)/batch_size))
			indices = np.random.randint(low=0, high=len(training_data), size=batch_size)
			#print("This is indices", np.shape(indices))
			target = [training_data[i][0] for i in indices]
			context = [embedding_avg(embedding_dim, training_data[i][1], word_to_ix) for i in indices]
			target_var = torch.Tensor(target)
			context_var = torch.stack(context)	
			#context_var, target_var = context_var.to(device), target_var.to(device)
			model.zero_grad()
			
			#print("this is context_var", context_var.size())
			sigmoid_outputs = model.forward(context_var)#.cuda()
			#sigmoid_outputs[sigmoid_outputs<=0.5] = 0
			#sigmoid_outputs[sigmoid_outputs>0.5] = 1
			sigmoid_outputs = sigmoid_outputs.double()
			target_var = target_var.double()#.cuda()
			#sigmoid_outputs = sigmoid_outputs.requires_grad_(True)
			
			loss = nn.BCELoss()
			#loss = nn.CrossEntropyLoss()
			#print("This is sigmoid_outputs size", sigmoid_outputs)
			#print("This is target_var size", target_var)
			my_loss = loss(sigmoid_outputs, target_var)
			my_loss.backward()
			optimizer.step()
		print("Epoch %d Loss: %.5f" % (epoch, my_loss.data))
	return model, vocab, word_to_ix

def main():
	# Loading and reading data for training and evaluation

	train_data = TextClassificationData('train.txt','./data')

	dev_data = TextClassificationData('dev.txt','./data')

	test_data = TextClassificationData('test.txt','./data')

	unlabelled_data = TextClassificationData('unlabelled.txt','./data')

	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	device = "cpu"

	num_labels =  2

	model_bow, vocab, word_to_ix = train(texts=train_data, model_type='bow', context_size=10, 
      embedding_dim=128, batch_size=256, num_epochs=10, lr=0.01, device=device)

	test_labels, test_vectors = data_to_tensor(test_data, vocab, word_to_ix, embedding_dim=128) 
	
	outputs = model_bow(test_vectors)
	outputs[outputs<=0.5] = 0
	outputs[outputs>0.5] = 1
	print(accuracy_calculator(outputs, test_labels))

	#dev_labels, dev_vectors = data_to_tensor(dev_data, vocab, word_to_ix, embedding_dim=128)
	#dev_outputs = model_bow(dev_vectors)
	#dev_outputs[dev_outputs<=0.5] = 0
	#dev_outputs[dev_outputs>0.5] = 1
	#print(accuracy_calculator(dev_outputs, dev_labels))	

	unlabelled_vectors = data_to_tensor(unlabelled_data, vocab, word_to_ix, embedding_dim=128, labelled=False)
	unlabelled_outputs = model_bow(unlabelled_vectors)
	unlabelled_outputs[dev_outputs<=0.5] = 0
	unlabelled_outputs[dev_outputs>0.5] = 1
	unlabelled_outputs = unlabelled_outputs.int()

	print(len(unlabelled_outputs.tolist()))
	print(set(unlabelled_outputs.tolist()))
	with open('predictions_q2.txt', 'w') as f:
		for item in unlabelled_outputs.tolist():
			f.write("%s\n" % item)

if __name__=="__main__":
	main()