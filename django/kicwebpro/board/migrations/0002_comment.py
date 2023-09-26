# Generated by Django 4.2.5 on 2023-09-14 09:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("board", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Comment",
            fields=[
                ("ser", models.AutoField(primary_key=True, serialize=False)),
                ("num", models.IntegerField(default=0)),
                ("content", models.CharField(max_length=4000)),
                ("regdate", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
