"""
Main program for COMS 6111 Advanced Database Project 1:
Information retrieval and query expansion using the Rocchio algorithm

Pitchapa Chantanapongvanij
Evan Ting I Lu

02/12/2021
"""


#############################################################################################
# Current Issues/To Do:
#   1. Performance could be improved (most relevant terms don't always get highest scores (?))
#   2. (bug) Should avoid adding repeated words (now will do so)
#############################################################################################


from pprint import pprint
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class SmartQuery():

    def __init__(self):
        self.API_key = "****"
        self.search_engine_ID = "****"
        self.current_query = ''
        self.relevant = []
        self.irrelevant = []
        self.corpus = []
        self.current_precision = 0.0

        # TF-IDF vectorizer params
        self._max_df = 0.95        # Ignore words with high df. (Similar effect to stopword filtering)
        self._min_df = 5           # Ignore words with low df.
        self._smooth_idf = True    # Smooth idf weights by adding 1 to df.
        self._sublinear_tf = True  # Replace tf with 1 + log(tf).

        # Rocchio params
        self._alpha = 1
        self._beta = 0.75
        self._gamma = 0.15

        # build Google API service object
        self.service = build("customsearch", "v1",
              developerKey=self.API_key) 

    def setter_query(self, new_query: str):

        self.current_query = new_query
        print('Query string updated to: \n', self.current_query)

    def initiate_query(self) -> dict:
        '''
        Start a query using Google JSON API
        query: user query string
        return query result in its raw format (JSON in Python dict format)
        '''

        # q: query and cx is our search engine ID 
        res = self.service.cse().list(
            q=self.current_query,
            cx=self.search_engine_ID,
        ).execute()

        return res

    def get_user_feedback(self, res: dict) -> float:
        '''
        Present the query results to user and record feedback and upate instance variables
        res: query result to present to user
        return current precision
        '''

        print("\nGoogle Search Results: ")
        print("======================\n\n")

        index = 0
        self.current_precision = 0.0

        for _ in res['items']:
            res_0 = res['items'][index]
            #print(type(res_0))
            #print(res_0.keys()) 

            ''' key info that we need '''
            url = res_0['formattedUrl']
            title = res_0['title']
            snippet = res_0['snippet']

            # insert this result into corpus i.e., whole document database
            self.corpus.insert(index, (title + ' ' + snippet))
            index += 1

            # display this result to user
            print('\nResult #{}\n[\nURL: {}\nTitle: {}\nSummary: {}\n]\n'
                .format(index, url, title, snippet))

            # Get relevance feedback and update relevant/irrelevant vectors
            while True:
                val = input("Relevant (Y/N) [Y]? ")
                # note: default to yes
                if val == '' or val in ['Y', 'y']:
                    self.relevant.append(index-1)
                    # ADD 0.1 to current percision count
                    self.current_precision += 0.1
                    break

                elif val in ['N', 'n']: 
                    self.irrelevant.append(index-1)
                    break
                
                else:
                    print('Please enter y/n\n')
                    continue
        
        return self.current_precision

    def update_query(self):
        '''
        update query string using the Rocchio algorithm
        '''

        # build vectorizer object with scikit-learn method
        vectorizer = TfidfVectorizer(max_df=self._max_df,
                                    smooth_idf=self._smooth_idf, sublinear_tf=self._sublinear_tf)

        # fit the vectorizer to whole corpus
        doc_tfidf = vectorizer.fit_transform(self.corpus).toarray()
        # convert query to vector space i.e., calculate tf-idf score for each term
        query_vec = vectorizer.transform([self.current_query]).toarray()

        rel_vecs = np.zeros(len(doc_tfidf[0]))
        for i in self.relevant:
            rel_vecs += doc_tfidf[i]
        rel_vecs /= len(doc_tfidf[0]) #normalization 

        irel_vecs = np.zeros(len(doc_tfidf[0]))
        for i in self.irrelevant:
            irel_vecs += doc_tfidf[i]
        irel_vecs /= len(doc_tfidf[0]) 

        # the Rocchio algorithm's updating formula
        new_query_vec = self._alpha * query_vec + self._beta * rel_vecs - self._gamma * irel_vecs

        # get the highest weigthing terms' corresponding indicies
        sorted_idx = np.argsort(new_query_vec) #sort from least -> most relevant weight of each term
        #print(sorted_idx)
        idx = sorted_idx[0][-1::-1] #reverse the array = most -> least relevant
        #print(idx)

        # print the score for each word in descending order
        # term_scores = dict(zip(vectorizer.get_feature_names_out(), new_query_vec[0].round(decimals=5)))
        # term_scores = {k: v for k, v in sorted(term_scores.items(), key=lambda item: item[1], reverse=True)}
        # print("term scores: ", term_scores)

        # Add 2 new words to query per iteration
        words = vectorizer.get_feature_names_out()

        count = 0
        index = 0
        while(len(idx)):
            top_index = idx[index]
            top_word = words[top_index]
            # print("count: ", count)
            if top_word not in self.current_query:
                new_query = self.current_query + " " + top_word
                self.setter_query(new_query=new_query)
                count +=1
            if count != 2:
                index +=1
            elif count == 2:
                break


        #CHECK if top1 and top2 are empty???
        #print("word 1 ", top_1, words[top_1])
        #print("word 2 ", top_2, words[top_2])

        #self.word_list.append(word2)

        #new_query = self.current_query + " " + word1 + " " + word2

def main():

    smart_query = SmartQuery()

    # see documentation of using build method to initiate a Google API:
    # https://edstem.org/us/courses/18455/discussion/1085971
    # First argument is the API name (we are using custom search API)
    # 2nd argument is version number
    # developerKey is our API key

    print("Parameters: " )
    print("Client key = ", smart_query.API_key)
    print("Engine key = ", smart_query.search_engine_ID)

    # get initial query strign and set precision
    initial_query = input("Please enter query: ")
    initial_query = initial_query.lower()
    smart_query.setter_query(initial_query)
    target_precision = float(input("Precision: "))

    while True:

        # initiate new query
        query_result = smart_query.initiate_query()
        precision = smart_query.get_user_feedback(res=query_result)

        # check if threshold is met
        if precision >= target_precision:
            break
        print("\ncurrent_precision: {:.2f}\n".format(precision))
        print("Updating query for you...\n\n")
        # update query string
        smart_query.update_query()

    if  smart_query.current_precision >= target_precision:
        # succeeded    
        print("======================")
        print("FEEDBACK SUMMARY")
        print("Query ", smart_query.current_query)
        print("Precision ", "{:.2f}".format(smart_query.current_precision))
        print("Desired precision reached, done\n")
    else:
        # failed
        print("We are sorry. Maybe try again with another query?\nThanks!")


if __name__ == "__main__":
    
    main()
