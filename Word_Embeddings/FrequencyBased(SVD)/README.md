# SVD method for creating embeddings

## <u>Files</u> 

* **SVD_utils.py** : Contains the code for creating embeddings using SVD
* **SVD.py** : Calls the SVD.py file and creates embeddings for the given dataset and saves it in a file

* **SVD_getnearestk.py** : Contains the code for finding the nearest neighbours of a given word using the embeddings created using SVD by using the embeddings saved in the file created by SVD_wrapper.py

<br>

## Dataset used for creating the embeddings
---


Stanford Movie Review Dataset : https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_10.json.gz

#### Sample Review of the data

```
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}
```

(After downloading the code locally, download and unzip the above dataset in the **Dataset** folder as the code)