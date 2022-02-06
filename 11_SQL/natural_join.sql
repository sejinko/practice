create table categories
(
category_id serial primary key,
category_name varchar (255) not null
);

create table products
(
product_id serial primary key,
product_name varchar (255) not null,
category_id int not null,
foreign key (category_id)
references categories (category_id)
);

INSERT INTO CATEGORIES
(CATEGORY_NAME)
VALUES
('Smart Phone')
, ('Laptop')
, ('Tablet')
;

INSERT INTO PRODUCTS
(PRODUCT_NAME, CATEGORY_ID)
VALUES
('iPhone', 1)
, ('Samsung Galaxy', 1)
, ('HP Elite', 2)
, ('Lenovo Thinkpad', 2)
, ('iPad', 3)
, ('Kindle Fire', 3)
;


select * from categories;

select 
*
from 
products a
natural join
categories b;


select 
	a.category_id, a.product_id,
	a.product_name, b.category_name
from
	products a
inner join
	categories b 
on (a.category_id = b.category_id);