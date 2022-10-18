from datetime import datetime
import csv
import sqlalchemy
from sqlalchemy import DateTime, PrimaryKeyConstraint, create_engine
from sqlalchemy import text
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import ForeignKey
from sqlalchemy import Date
from sqlalchemy import Float
from sqlalchemy import select
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session
from sqlalchemy.orm import declarative_base
from tqdm import tqdm

Base = declarative_base()


class Sub(Base):
    __tablename__ = "sub"
    adsh = Column(String, nullable=False, primary_key=True)
    cik = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    sic = Column(Integer)
    countryba = Column(String)
    stprba = Column(String)
    cityba = Column(String)
    zipba = Column(String)
    bas1 = Column(String)
    bas2 = Column(String)
    baph = Column(String)
    countryma = Column(String)
    stprma = Column(String)
    cityma = Column(String)
    zipma = Column(String)
    mas1 = Column(String)
    mas2 = Column(String)
    countryinc = Column(String)
    stprinc = Column(String)
    ein = Column(Integer)
    former = Column(String)
    changed = Column(String)
    afs = Column(String)
    wksi = Column(String, nullable=False)
    fye = Column(String)
    form = Column(String, nullable=False)
    period = Column(Date)
    fy = Column(String)
    fp = Column(String)
    filed = Column(Date, nullable=False)
    accepted = Column(DateTime, nullable=False)
    prevrpt = Column(String, nullable=False)
    detail = Column(String, nullable=False)
    instance = Column(String, nullable=False)
    nciks = Column(Integer, nullable=False)
    aciks = Column(String)
    def __repr__(self):
        return f'Sub(adsh={self.adsh}, cik={self.cik}, form={self.form}, period={self.period}'

class Tag(Base):
    __tablename__ = 'tag'
    tag = Column(String, nullable=False, primary_key=True)
    version = Column(String, nullable=False, primary_key=True)
    custom = Column(String, nullable=False)
    abstract = Column(String, nullable=False)
    datatype = Column(String)
    iord = Column(String)
    crdr = Column(String)
    tlabel = Column(String)
    doc = Column(String)
    def __repr__(self):
        return f'Tag(tag={self.tag}'

class Num(Base):
    __tablename__ = "num"
    id = Column(Integer, primary_key=True)
    adsh = Column(String, ForeignKey('sub.adsh'), nullable=False)
    tag = Column(String, nullable=False)
    version = Column(String, nullable=False)
    coreg = Column(String)
    ddate = Column(Date, nullable=False)
    qtrs = Column(Integer, nullable=False)
    uom = Column(String, nullable=False)
    value = Column(Float)
    footnote = Column(String)
    def __repr__(self):
        return f"Num(adsh={self.adsh}, tag={self.tag}, ddate={self.ddate}, value={self.value}"

engine = create_engine("postgresql://postgres:12345678@localhost/stocks", echo=True)
conn = engine.raw_connection()
Base.metadata.create_all(engine)
session = Session(engine)
cursor = conn.cursor()  
session.execute(f"CREATE TABLE temp_tag (LIKE tag INCLUDING DEFAULTS)")
foldernames = [f'{year}q{q}' for year in range(2009, 2022) for q in range(1, 5)]
for folder in tqdm(foldernames):
    with open(f'{folder}/sub.csv', 'r') as f:
        cmd = f"copy sub(adsh, cik, name, sic, countryba, stprba, cityba, zipba, bas1, bas2, baph, countryma, stprma, cityma, zipma, mas1, mas2, countryinc, stprinc, ein, former, changed, afs, wksi, fye, form, period, fy, fp, filed, accepted, prevrpt, detail, instance, nciks, aciks) from STDIN with (delimiter '\t', format csv, header true, quote '\b')"
        cursor.copy_expert(cmd, f)
        conn.commit()
    print(f'{folder}/sub done')
    with open(f'{folder}/tag.csv', 'r') as f:
        cmd = f"copy temp_tag(tag, version, custom, abstract, datatype, iord, crdr, tlabel, doc) from STDIN with (delimiter '\t', format csv, header true, quote '\b')"
        cursor.copy_expert(cmd, f)
        conn.commit()
    print(f'{folder}/tag done')
    cursor.execute( f"INSERT INTO tag SELECT * FROM temp_tag ON CONFLICT DO NOTHING")
    cursor.execute(f"TRUNCATE temp_tag")
    conn.commit()
    with open(f'{folder}/num.csv', 'r') as f:
        cmd = f"copy num(adsh, tag, version, coreg, ddate, qtrs, uom, value, footnote) from STDIN with (delimiter '\t', format csv, header true, quote '\b')"
        cursor.copy_expert(cmd, f)
        conn.commit()
    print(f'{folder}/num done')