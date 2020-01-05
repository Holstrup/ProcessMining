import xml.etree.ElementTree as ET

class ExtractDataLog:

    def __init__(self,path):
        self.tree = ET.parse(path)

    def create_dict(self):
        thread_dict={'Thread ID':0 ,'Posts':{}}
        root = self.tree.getroot()

        #Thread ID
        thread_dict['Thread ID'] = root[0].text

        #Absolute position
        abs_pos = 1

        #initial post
        thread_dict['Posts'][abs_pos]={}

        for init_post in root.findall('InitPost'):
            user_id = init_post.find('UserID').text
            thread_dict['Posts'][abs_pos]['User ID'] = user_id

            date = init_post.find('Date').text
            thread_dict['Posts'][abs_pos]['Date'] = date

            content = init_post.find('icontent').text
            thread_dict['Posts'][abs_pos]['Content'] = content

            post_class=init_post.find('Class').text
            thread_dict['Posts'][abs_pos]['Class'] = post_class

        #rest of posts in thread
        for post in root.findall('Post'):

            abs_pos += 1
            thread_dict['Posts'][abs_pos] = {}

            user_id = post.find('UserID').text
            thread_dict['Posts'][abs_pos]['User ID'] = user_id

            date = post.find('Date').text
            thread_dict['Posts'][abs_pos]['Date'] = date

            content = post.find('rcontent').text
            thread_dict['Posts'][abs_pos]['Content'] = content

            post_class = post.find('Class').text
            thread_dict['Posts'][abs_pos]['Class'] = post_class

        return thread_dict
