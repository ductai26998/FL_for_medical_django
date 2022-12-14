# Generated by Django 3.2.12 on 2022-12-05 16:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('device', '0002_auto_20221203_0002'),
    ]

    operations = [
        migrations.AddField(
            model_name='device',
            name='val_acc',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='device',
            name='val_acc_list',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='device',
            name='val_loss',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='device',
            name='val_loss_list',
            field=models.TextField(blank=True, null=True),
        ),
    ]
