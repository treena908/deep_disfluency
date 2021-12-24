
import csv
db_set=set()
swdb_set=set()
with open('tag_representations/DB_disf1_tags.csv') as db:
    DB=csv.reader(db)


    for line in DB:


        db_set.add(line[1])
    db.close()
with open('tag_representations/swbd_disf1_041_tags.csv') as db:
    DB = csv.reader(db)

    for line in DB:
        swdb_set.add(line[1])
    db.close()
print(db_set.difference(swdb_set))

print(len(db_set.difference(swdb_set)))