from transformers import BartForConditionalGeneration, BartTokenizer
import speech_recognition as sr
# install libraries: pyTorch, TensorFlow, Flax, Transformers and SpeechRecognition

class GenMOM:

    def __init__(self):
        self.ip = ""
        self.op = ""

    def speechToText(self):
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                print("Please start speaking...")
                audio2 = r.listen(source2, 10, 60) # runs for 60s
                print("Converting speech to text...")
                self.ip = r.recognize_google(audio2)
                print("Orignal: ", self.ip)
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            print("unknown error occurred")

    def summarizeWithBart(self):
        print("Summarising text...")
        model_name = 'facebook/bart-large-cnn'
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        inputs = tokenizer([self.ip], return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=150, early_stopping=True)
        self.op = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


mom = GenMOM()
mom.speechToText()
mom.summarizeWithBart()
print("MoM: ", mom.op)

# input speech
"""There are many techniques available to generate extractive summarization. To keep it simple, I will be using an 
unsupervised learning approach to find the sentences similarity and rank them. Summarization can be defined as the 
task of producing a concise and fluent summary, while preserving key information and overall meaning. One benefit of 
this is that, you don’t need to train and build a model prior to start using it for your project. It’s good to 
understand Cosine similarity to make the best use of the code you are going to see. Cosine similarity is a measure of 
similarity between two non-zero vectors of an inner product space, that measures the cosine of the angle between 
them. The angle will be 0 if sentences are similar. """

# speech to text:
"""there are many techniques available to generate extractive summarization to keep it simple I will be 
using an unsupervised learning approach to find the sentence similarity and rank them summarization can be defined as 
the task of producing a concise and fluent summary while preserving key information and overall meaning one benefit 
of this is that you don't need train and build a model prior to start using it for your project it's good to 
understand similarity to make the best use of the courier going to see cosine similarity is a measure of similarity 
between two non zero vectors of an inner product space that measures the cosine of the angle between them the angle 
will be zero at the sentences as similar """

# output MoM
"""There are many techniques available to generate extractive summarization. I will be using an unsupervised 
learning approach to find the sentence similarity and rank them summarization can be defined as the task of producing 
a concise and fluent summary while preserving key information. """
