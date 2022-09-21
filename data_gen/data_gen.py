from collections import defaultdict
import pandas as pd


class DataGen(object):

    def __init__(self,df):

        self.dataframe = df

        self._word_dict = defaultdict()
        self._label_dict = set([])

        self.slotFrame = pd.DataFrame({'Sentence':[],"Slots":[]})

    def generate_sequences(self,count = 100):
    
        sentences = self.dataframe['trans_input'].values
        all_slots = []

        cur_sents = sentences[:count+1]

        for sent in cur_sents:
            slots = []
            words = sent.split(" ")
            for w in words:
                if w not in self._word_dict:
                    print(w)
                    intent = int(input("Give intent in 1. I-SUB, 2. I-Field 3. Others"))
                    self._word_dict[w] = intent
                    slots.append(intent)
                else:
                    slots.append(self._word_dict[w])
            all_slots.append(slots)
           

        self.slotFrame['Slots'] = all_slots
        self.slotFrame['Sentence'] = cur_sents

        self.slotFrame.to_csv("slot_filled.csv")

            
if __name__ == "__main__" :

    df  = pd.read_csv("slot_filling_df.csv")
    dg = DataGen(df)
    dg.generate_sequences()
