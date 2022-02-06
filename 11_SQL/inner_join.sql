CREATE TABLE BASKET_A (
ID INT PRIMARY KEY
, FRUIT VARCHAR
(100) NOT NULL
)
;

select * from basket_a;

CREATE TABLE BASKET_B (
ID INT PRIMARY KEY
, FRUIT VARCHAR
(100) NOT NULL
)
;

select * from basket_b;

INSERT INTO BASKET_A (ID, FRUIT)
VALUES (1, 'Apple'), (2, 'Orange'), (3, 'Banana'), (4, 'Cucumber'
)
;

COMMIT;

INSERT INTO BASKET_B (ID, FRUIT)
VALUES (1, 'Orange'), (2, 'Apple'), (3, 'Watermelon'), (4, 'Pear') ;

COMMIT; 


select 
	a.id id_a,
	a.fruit fruit_a,
	b.id id_b,
	b.fruit fruit_b
from basket_a a
inner join basket_b b 
on a.fruit = b.fruit;


select
	a.customer_id,
	a.first_name,
	a.last_name,
	a.email,
	b.amount,
	b.payment_date
from customer a
inner join payment b 
on a.customer_id = b.customer_id;


select 
	a.customer_id,
	a.first_name,
	a.last_name,
	a.email,
	b.amount,
	b.payment_date,
	c.first_name as s_first_name,
	c.last_name as s_last_name
from
	customer a
inner join payment b 
on a.customer_id = b.customer_id 
inner join staff c 
on b.staff_id = c.staff_id;
	
	