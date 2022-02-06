CREATE TABLE PRODUCT_GROUP (
GROUP_ID SERIAL PRIMARY KEY,
GROUP_NAME VARCHAR (255) NOT NULL
);

CREATE TABLE PRODUCT (
PRODUCT_ID SERIAL PRIMARY KEY
, PRODUCT_NAME VARCHAR (255) NOT NULL
, PRICE DECIMAL (11, 2)
, GROUP_ID INT NOT NULL
, FOREIGN KEY (GROUP_ID)
REFERENCES PRODUCT_GROUP (GROUP_ID)
);

INSERT INTO PRODUCT_GROUP (GROUP_NAME)
VALUES
('Smartphone')
, ('Laptop')
, ('Tablet');

INSERT INTO PRODUCT (PRODUCT_NAME,
GROUP_ID,PRICE)
VALUES
('Microsoft Lumia', 1, 200)
, ('HTC One', 1, 400)
, ('Nexus', 1, 500)
, ('iPhone', 1, 900)
, ('HP Elite', 2, 1200)
, ('Lenovo Thinkpad', 2, 700)
, ('Sony VAIO', 2, 700)
, ('Dell Vostro', 2, 800)
, ('iPad', 3, 700)
, ('Kindle Fire', 3, 150)
, ('Samsung Galaxy Tab', 3, 200);


select
	*
from
	product;
	

select 
	count(*)
from
	product;


select 
	count(*) over(), A.*
from
	product A;


select 
	avg(price)
from
	product;


select 
	b.group_name,
	avg(price)
from
	product a
inner join product_group b 
on (a.group_id = b.group_id)
group by 
b.group_name;


select 
	a.product_name,
	a.price,
	b.group_name,
	avg(a.price) over(partition by b.group_name)
from 
	product a
inner join product_group b 
on (a.group_id = b.group_id);




