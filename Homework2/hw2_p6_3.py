import numpy as np
import time
import os
import copy
import re
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import more_itertools as mit

torch.manual_seed(1)

class TextClassificationData(Dataset):
	"""
	Dataset object considering the data for text classification
	"""

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
		
	def __len__(self):
		return len(self.txt_data)

	def __getitem__(self, idx):
		phrase = self.txt_data[idx][2:-3]
		label = self.txt_data[idx][0]
		sample = {'Label': label, 'phrase': phrase}
		return sample

def pad(l, size, padding):
	return l + [padding] * abs((len(l)-size))

def gather_word_freqs(raw_data, min_count = 7, max_length=18):
	"""
	Function that will create a vocabulary and a dictionary of word -index during training.
	This function will only be used during training. 
	"""
	print('building vocabulary ...')
	tmp_vocab = {}
	ix_to_word = {}
	word_to_ix = {}
	total = 0.0
	
	for doc in raw_data:
		#print(doc)
		phrase = list(doc.values())[1].split()
		#print(phrase)
		#phrase = re.sub(r'[^\w\s]','',phrase)
		#phrase = doc.split()[1:]
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
	labels_list = []
	max_len_sentence = 0
	vector_sentence = []
	for doc in raw_data:
		phrase = list(doc.values())[1].split()
		#phrase = doc.split()[1:]
		label = list(doc.values())[0]
		new_phrase = []
		tmp_vector_sentence = []
		for i, word in enumerate(phrase):
			if word in vocab:
				new_phrase.append(word)
				tmp_vector_sentence.append(word_to_ix[word])
		if len(tmp_vector_sentence) < max_length:
			tmp_vector_sentence = pad(tmp_vector_sentence, max_length, 0)
		vector_sentence.append(tmp_vector_sentence)
		new_split_text.append(new_phrase)
		labels_list.append(int(label))
	print('vocabulary size', len(vocab.keys()), len(word_to_ix), len(ix_to_word))
	return new_split_text, labels_list, vector_sentence, vocab, word_to_ix, ix_to_word, max_len_sentence


class WordEmbeddingAvgPooling(nn.Module):

	def __init__(self, vocab_size, embedding_dim, n_out):
		super().__init__()
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.n_out = n_out
		#self.context_size = context_size
		self.embeddings = nn.Embedding(vocab_size, embedding_dim, scale_grad_by_freq=True)
		#self.avg_pooling = nn.AvgPool1d(3, stride=1)
		self.linear = nn.Linear(embedding_dim, n_out)
		
        	
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


def embedding_avg(embedding_dim, list_words, word_to_ix, glove=False):
	num_vocab = len(word_to_ix)
	embedding = nn.Embedding(num_vocab, embedding_dim)
	total = torch.zeros([1, embedding_dim], dtype=torch.float)
	for w in list_words:
		if glove is False:
			curr_vector = torch.tensor([word_to_ix[w]], dtype=torch.long)
		else:
			curr_vector = torch.tensor(word_to_ix[w], dtyp=torch.long)
		embedded = embedding(curr_vector)
		total = total + embedded
	avg_embedded = torch.div(total,len(list_words))
	return avg_embedded


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

def train(texts, labels, word_to_ix, model_type, context_size, embedding_dim, batch_size, num_epochs, lr, device, glove=False):
	training_data = texts
	num_labels=2

	# build the model and optimizer
	model = WordEmbeddingAvgPooling(len(word_to_ix), embedding_dim, num_labels)
	
	#optimizer = optim.SGD(model.parameters(), lr=lr)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	print(device)
	print(model)
	#model.to(device)
    # train the mo
	
	print("Starting training")
	for epoch in range(num_epochs):
		print("This is epoch ", epoch)
		for iteration in range(int(len(word_to_ix)/batch_size)):
			indices = np.random.randint(low=0, high=len(training_data), size=batch_size)
			target = [labels[i] for i in indices]
			#curr_phrases = [avg_vector(training_data[i]) for i in indices]

			context = [embedding_avg(embedding_dim, training_data[i], word_to_ix, glove) for i in indices]
			#context = [vectors[i] for i in indices]
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
	return model, word_to_ix


def main():
	# Loading and reading data for training and evaluation

	sample = TextClassificationData('train.txt','./data')
	train_text, train_labels, vector_sentence, train_vocab, train_word_to_ix, ix_to_word, max_len_sentence = gather_word_freqs(sample)	

	embeddings_index = dict()
	f = open('glove.6B.100d.txt', encoding="utf8")
	for line in f:
		#print(type(line))
		#print(line.split()[0])
		#print(np.array(line.split()[1:]).astype(np.float))
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	#	print(embeddings_index)
	f.close()
	
	#print(embeddings_index)

	#print(train_text)
	#train_text, train_labels, vector_sentence, train_vocab, train_word_to_ix, ix_to_word, max_len_sentence = TextClassificationData('train.txt','./data').gather_word_freqs()
	#train_text, train_labels, vector_sentence, train_vocab, train_word_to_ix, ix_to_word, max_len_sentence = TextClassificationData('dev.txt','./data').gather_word_freqs()
	
	#test_text, test_labels, test_vector_sentence, test_vocab, test_word_to_ix, ix_to_word, max_len_sentence = TextClassificationData('test.txt','./data').gather_word_freqs()

	#unlabelled_data = TextClassificationData('unlabelled.txt','./data')

	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	device = "cpu"

	num_labels =  2

	model_bow, word_to_ix = train(train_text, train_labels, embeddings_index, model_type='bow', context_size=10, 
      embedding_dim=128, batch_size=256, num_epochs=10, lr=0.01, device=device, glove=True)

	#test_labels, test_vectors = data_to_tensor(test_data, vocab, word_to_ix, embedding_dim=128) 
	
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
	with open('predictions_q3.txt', 'w') as f:
		for item in unlabelled_outputs.tolist():
			f.write("%s\n" % item)

if __name__=="__main__":
	main()