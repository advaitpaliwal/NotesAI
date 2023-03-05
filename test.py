import torch
from transformers import BertTokenizer, BertForSequenceClassification


class BERTNoteGenerator:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    def generate_notes(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        labels = torch.argmax(probabilities, dim=1)
        notes = []
        for i, label in enumerate(labels):
            if label == 1:
                notes.append(inputs['input_ids'][i])
        generated_notes = self.tokenizer.batch_decode(notes, skip_special_tokens=True)
        return generated_notes
chunks = """
hi everybody it's me waddles diamonds are your best friend or you want them to be no matter what your current situation is if you like diamonds this is the video for you with the 1.19 update diamond mining has changed again in this video we're going to take a look at diamond mining and talk about some of the best ways to do it before we get too busy make a quick stop at the like button and if you're new here hi welcome if you'd like to become genius then subscribing it's a great idea everybody knows what a diamond is so fortunately we can skip that check out this chart right here this is the current or generation chart as of minecraft 1.19 this chart is our first big guide when it comes to mining diamonds efficiently broken down into words and simple words of that diamond generation how does it work well the diamonds generate in multiple batches both on java and betarock edition diamonds are going to get more common the deeper you go in your world when you take a look at bedrock and java edition side by side there are actually some minor differences between diamond generation one difference that doesn't exist is the fact that most diamonds or many of them at least are going to generate with no air exposure meaning you're gonna have to dig to find them you know how copper ore generates more commonly inside the drips on caves bomb well diamonds aren't like that they have no connection to biome at all diamond ore generates at world generation when diamond ore generates it can actually replace a couple blocks in the game diamond ore will replace stone deepside granite andesite diorite and tough if it's in the way unless you're on bedrock edition on bedrock tough can't be replaced by diamond ore
"""
print(BERTNoteGenerator().generate_notes(chunks))