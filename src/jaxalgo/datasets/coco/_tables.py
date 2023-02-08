"""Tables.

P.S.

without explicitly adding primary key tag
sqlite will use rowid as the auto-increase primary key
"""
from sqlalchemy import INTEGER
from sqlalchemy import REAL
from sqlalchemy import TEXT
from sqlalchemy import Column
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import Table

metadata = MetaData()

images = Table(
    "f_image",
    metadata,
    Column("imageid", Integer, index=True),
    Column("name", TEXT, index=True, unique=True),
    Column("height", INTEGER),
    Column("width", INTEGER),
    Column("url", TEXT),
    Column("path", TEXT),
)

cates = Table(
    "d_cate",
    metadata,
    Column("cateid", Integer, index=True),
    Column("name", TEXT),
)

boxes = Table(
    "f_box",
    metadata,
    Column("boxid", Integer, index=True),
    Column(
        "imageid",
        Integer,
        ForeignKey("f_image.imageid", onupdate="CASCADE", ondelete="CASCADE"),
    ),
    Column(
        "cateid",
        Integer,
        ForeignKey("d_cate.cateid", onupdate="CASCADE", ondelete="CASCADE"),
    ),
    Column("bbox1", REAL),
    Column("bbox2", REAL),
    Column("bbox3", REAL),
    Column("bbox4", REAL),
)

labels = Table(
    "f_yolo_label",
    metadata,
    Column("boxid", Integer),
    Column("imageid", Integer),
    Column("cateid", Integer),
    Column("x", REAL),
    Column("y", REAL),
    Column("w", REAL),
    Column("h", REAL),
    Column("cate_name", TEXT),
    Column("image_name", TEXT),
)
