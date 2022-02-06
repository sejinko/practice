CREATE TABLE SALES
(
BRAND VARCHAR NOT NULL,
SEGMENT VARCHAR NOT NULL,
QUANTITY INT NOT NULL,
PRIMARY KEY (BRAND, SEGMENT)
);

INSERT INTO SALES (BRAND, SEGMENT, QUANTITY) VALUES
('ABC', 'Premium', 100)
, ('ABC', 'Basic', 200)
, ('XYZ', 'Premium', 100)
, ('XYZ', 'Basic', 300);

select * from sales

select
	brand,
	segment,
	sum(quantity)
from
	sales
group by
	brand,
	segment;
	

select 
	brand,
	sum(quantity)
from
	sales
group by
	brand;
	

select 
	segment,
	sum(quantity)
from
	sales
group by segment;


select 
	sum(quantity)
from sales;


select
	brand,
	segment,
	sum(quantity)
from sales
group by brand, segment 
union all 
select 
	brand,
	null,
	sum(quantity)
from sales
group by segment 
union all 
select
	null,
	null,
	sum(quantity)
from
	sales;




select 
	brand,
	segment,
	sum(quantity)
from
	sales
group by
grouping sets 
((brand, segment), (brand), (segment),());


select 
	grouping(brand) grouping_brand,
	grouping(segment) grouping_segment,
	brand,
	segment,
	sum(quantity)
from
	sales
group by
grouping sets 
(
	(brand, segment),
	(brand),
	(segment),
	()
)
order by brand, segment asc;

	
