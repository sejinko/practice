CREATE TABLE SALES2007_1
(
NAME VARCHAR(50)
, AMOUNT NUMERIC(15,2)
);

INSERT INTO SALES2007_1
VALUES
('Mike', 150000.25)
, ('Jon', 132000.75)
, ('Mary', 100000)
;

CREATE TABLE SALES2007_2
(
NAME VARCHAR(50)
, AMOUNT NUMERIC(15,2)
);

INSERT INTO SALES2007_2
VALUES
('Mike', 120000.25)
, ('Jon', 142000.75)
, ('Mary', 100000)
;


select * from sales2007_1;

select * from sales2007_2;



select 
	*
from
	sales2007_1
union
select
	*
from 
	sales2007_2;


select 
	*
from 
	sales2007_1
union
select
	*
from 
	sales2007_2 s 
order by amount desc;




