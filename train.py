import sys
sys.path.append('..')
from optimizer import Adam
from cbow import SimpleCBOW, CBOW
from util import preprocess, create_contexts_target, convert_one_hot
from trainer import Trainer

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus)

vocab_size=len(word_to_id)
target=convert_one_hot(target, vocab_size)
contexts=convert_one_hot(contexts, vocab_size)

#model = SimpleCBOW(vocab_size, hidden_size)
model = CBOW(corpus, vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()