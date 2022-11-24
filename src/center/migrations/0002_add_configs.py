from django.db import migrations
from src.center import CenterConfigKey


def apply(apps, schema_editor):
    CenterConfig = apps.get_model("center", "CenterConfig")
    db_alias = schema_editor.connection.alias
    new_objects = []
    existed_names = (
        CenterConfig.objects.using(db_alias)
        .filter(
            name__in=[
                CenterConfigKey.EPOCHS,
            ]
        )
        .values_list("name", flat=True)
    )
    if CenterConfigKey.EPOCHS not in existed_names:
        new_objects.append(
            CenterConfig(
                name=CenterConfigKey.EPOCHS,
                value="10",
                value_type="int",
            )
        )
    CenterConfig.objects.using(db_alias).bulk_create(new_objects)


def rollback(apps, schema_editor):
    CenterConfig = apps.get_model("center", "CenterConfig")
    db_alias = schema_editor.connection.alias
    CenterConfig.objects.using(db_alias).filter(
        name__in=[
            CenterConfigKey.EPOCHS,
        ]
    ).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("center", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(apply, rollback),
    ]
