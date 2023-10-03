from peewee import *

db = SqliteDatabase('chatbot.db')


class ChatHistory(Model):
    id = AutoField(primary_key = True)
    query = TextField()
    model = TextField()
    answer = TextField()
    millisecs = IntegerField()

    class Meta:
        database = db # This model uses the "chatbot.db" database.


db.connect()
db.create_tables([ChatHistory])