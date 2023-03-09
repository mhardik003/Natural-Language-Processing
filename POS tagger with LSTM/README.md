# LSTM based POS Tagger 

**Dataset used** : UD_English-Atis from <a href="https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4923">here</a>

---

## File structure

- **UD_English-Atis** (contains the data)
- **Encoding_Dictionaries**
    - **idx2tag.pkl, words2idx.pk**l (Dictionaries for storing the encoding for the tags and the words)
- **lstm.ipynb** (python notebook version of model.py)
- **model.py** (Contains the main model)
- **pos_tagger_pretrained_model.pt** (the pytorch pretrained model created and saved on running model.py)
- **pos_tagger.py** (program that loads the pretrained model, idx2tag.pkl , words2idx.pkl , ask for an input sentence and gives it corresponding tags)

---

## To run the files
* To run model.py (which contains the main model)
    ```
    python3 model.py
    ```

* To run the pos_tagger.py
    ```
    python3 pos_tagger.py
    ```




