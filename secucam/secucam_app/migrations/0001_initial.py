# Generated by Django 3.0.3 on 2024-03-09 12:20

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Footage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('footage_name', models.CharField(max_length=200)),
            ],
        ),
    ]
