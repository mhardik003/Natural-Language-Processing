from SVD_utils import *

# def get_input(s):
#     """
#     Function to ensure that the user enters an integer value
#     """

#     flag = 0
#     print(s, end=" ")
#     while (flag != 1):
#         try:
#             val = int(input())
#             flag = 1
#         except ValueError:
#             print("Please enter an integer value")
#             flag = 0

#     return val


# NUM_REVIEWS = get_input("> Enter the number of reviews to use :")
# EMBEDDING_SIZE = get_input("> Enter the embedding size :")
# WINDOW_SIZE = get_input("> Enter the window size :")
# NUM_MOST_COMMON_WORDS_TO_IGNORE = get_input(
#     "> Enter the number of most common words to ignore :")
# FREQ_UNDER_WHICH_TO_IGNORE = 4

EMBEDDING_SIZE = 100
filename = "../Dataset/Movies_and_TV_10.json"
word_list = ['horror', 'good', 'kill', 'hero', 'action']


encoder = COMatrixEncoder(filename)
encoder.create_cooccurrence_matrix()
encoder.create_embeddings(EMBEDDING_SIZE)
encoder.plot_embeddings(word_list)
encoder.save_data()
