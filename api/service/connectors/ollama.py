from ollama import Client
from pydantic import BaseModel

"""id,target,comment_text,severe_toxicity,obscene,identity_attack,insult,threat,asian,atheist,bisexual,black,buddhist,christian,female,heterosexual,hindu,homosexual_gay_or_lesbian,intellectual_or_learning_disability,jewish,latino,male,muslim,other_disability,other_gender,other_race_or_ethnicity,other_religion,other_sexual_orientation,physical_disability,psychiatric_or_mental_illness,transgender,white,funny,wow,sad,likes,disagree,sexual_explicit,identity_annotator_count"""

SYSTEM_PROMPT = """
You are a helpful assistant and content moderation model that evaluates the toxicity of a message. Your goal is to return a JSON object with float scores (ranging from 0.0 to 1.0) indicating the severity of the following categories:

obscene: Use of offensive or profane language.
identity_attack: Attacks or slurs directed at someone's identity (e.g., race, gender, religion).
insult: Personal attacks or disrespectful language.
threat: Statements suggesting intent to cause harm or violence.
Your scoring should reflect the severity and intent of the message. Mild or informal rudeness (e.g., "you are stupid") should receive low scores, whereas clearly harmful, threatening, or hateful language should receive high scores.

Use the following guidelines:

Reserve scores above 0.7 for severely toxic or harmful content.
Mildly offensive or disrespectful language should typically score below 0.3.
Neutral or benign messages should score close to 0.0.

Be as accurate as possible in your classification. The input will be a single message in plain text.
"""

class ResponseFormat(BaseModel):
    # target: float
    obscene: float
    identity_attack: float
    insult: float
    threat: float

class Ollama:
    def __init__(self, model: str = "llama3.2:1b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.client = Client(host=base_url)

    def get_response(self, message: str):
        res = self.client.chat(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"INPUT - {message}"
                }
            ],
            model=self.model,
            format=ResponseFormat.model_json_schema()
        )

        response = ResponseFormat.model_validate_json(res.message.content)

        return response

