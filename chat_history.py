from peewee import *

db = SqliteDatabase('chatbot.db')


class BaseModel(Model):
    class Meta:
        database = db # This model uses the "chatbot.db" database.


class ChatHistory(BaseModel):
    id = AutoField(primary_key=True)
    query = TextField()
    model = TextField()
    answer = TextField()
    millisecs = IntegerField()


class ChatBenchmark(BaseModel):
    id = AutoField(primary_key=True)
    query = TextField()
    model_a = TextField()
    model_b = TextField()
    answer_a = TextField()
    answer_b = TextField()
    time_a = IntegerField()
    time_b = IntegerField()
    winner = TextField()
    judge = TextField()


db.connect()
db.create_tables([ChatHistory])
db.create_tables([ChatBenchmark])
