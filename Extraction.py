import json

class Extract:
    all_threads_list: list
    def __init__(self,dataset):

        if dataset == 'MSDialog':
            self.extract_ms()


    def extract_ms(self):
        """Dict has following structure:
                {'Thread ID': Str, Posts: {abs_pos(int): {'User id': Str,  'Content': Str}}}

            all_threads_list: List containing all thread dictionaries.
                         """
        self.all_threads_list=[]

        # read file
        with open('MSDialog/Intent/MSDialog-Intent.json', 'r') as myfile:
            data = myfile.read()

        # parse file
        all_threads = json.loads(data)

        for thread in all_threads:
            thread_dict={}

            thread_id = thread
            thread_dict['Thread ID']=thread_id
            thread_dict['Posts'] = {}



            for post in all_threads[thread_id]['utterances']:

                abs_pos =post['utterance_pos']
                user_id=post['user_id']
                content=post['utterance']

                thread_dict['Posts'][abs_pos] = {}
                thread_dict['Posts'][abs_pos]['User ID'] = user_id
                thread_dict['Posts'][abs_pos]['Content'] = content
            self. all_threads_list.append(thread_dict)

    def get_all_threads_list(self):

        return self.all_threads_list




