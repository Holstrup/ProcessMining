import random
import os
import json
import xml.etree.ElementTree as ET


class Data_Processing_Transformation:
    def __init__(self, path_name):
        self.path_name = path_name             # Slack filename
        self.message_types = ["Q", "A", "C"] # Add more types if needed

        # Run Pipeline
        self.extraction()
        self.transformation()


    def extraction(self):
        directory = os.listdir(self.path_name)
        directory.sort()
        self.data = []
        for filename in directory:
            if filename.endswith(".DS_Store"):
                continue
            else:
                with open(os.path.join(self.path_name, filename), 'r') as json_file:
                    self.data.extend(json.load(json_file))


    def transformation(self):
        ts_start = 0.0
        convo_id = 0
        root = ET.Element('log')
        # <log xes.version="1.0" xes.features="nested-attributes" openxes.version="1.0RC7" xmlns="http://www.xes-standard.org/">
        root.set("xes.version", "1.0")
        root.set("xes.features", "nested-attributes")
        root.set("openxes.version", "1.0RC7")
        root.set("xmlns", "http://www.xes-standard.org/")

        for message in self.data:
            if 'subtype' in message.keys():
                continue
            else:
                message['classification'] = self.classification(message['text'])
                if ts_start == 0:
                    ts_start = float(message['ts'])
                    ConversationElement = ET.SubElement(root, "trace")
                    ConversationElement.tail = "\n"  # Edit the element's tail
                    ConversationElement.text = ""

                    caseIdElement = ET.SubElement(ConversationElement, "string")
                    caseIdElement.set("key", "id")
                    caseIdElement.set("value", str(convo_id))
                    caseIdElement.tail = "\n"  # Edit the element's tail
                    caseIdElement.text = ""

                elif abs(ts_start - float(message['ts'])) > 5500.0: # Split roughly every 1,5 hours
                    convo_id += 1
                    ts_start = float(message['ts'])

                    ConversationElement = ET.SubElement(root, "trace")
                    ConversationElement.tail = "\n"  # Edit the element's tail
                    ConversationElement.text = ""

                    caseIdElement = ET.SubElement(ConversationElement, "string")
                    caseIdElement.set("key", "id")
                    caseIdElement.set("value", str(convo_id))
                    caseIdElement.tail = "\n"  # Edit the element's tail
                    caseIdElement.text = ""

                MessageElement = ET.SubElement(ConversationElement, "event")
                nameElement = ET.SubElement(MessageElement, "string")

                nameElement.set("key", "name")
                nameElement.set("value", message['user_profile']['real_name'])
                nameElement.tail = "\n"  # Edit the element's tail
                nameElement.text = ""

                nameElement = ET.SubElement(MessageElement, "string")

                nameElement.set("key", "message_content")
                nameElement.set("value", message['text'])
                nameElement.tail = "\n"  # Edit the element's tail
                nameElement.text = ""

                nameElement = ET.SubElement(MessageElement, "string")

                nameElement.set("key", "classification")
                nameElement.set("value", message['classification'])
                nameElement.tail = "\n"  # Edit the element's tail
                nameElement.text = ""

                nameElement = ET.SubElement(MessageElement, "date")

                nameElement.set("key", "timestamp")
                nameElement.set("value", message['ts'])
                nameElement.tail = "\n"  # Edit the element's tail
                nameElement.text = ""

                MessageElement.tail = "\n"  # Edit the element's tail
                MessageElement.text = "\n"  # Edit the element's tail


        tree = ET.ElementTree(root)
        tree.write("data.xes", encoding="UTF-8", xml_declaration=True)


    def classification(self, message):
        """
        This is where the magic happens. We need to take a message in
        and output a classification based on the message

        @:param message: Message String
        @:return classifiction: Q, A or C
        """

        return random.choice(self.message_types)


