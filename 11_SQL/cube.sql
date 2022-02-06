select 
	brand,
	segment,
	sum(quantity)
from
	sales
group by
	cube(brand, segment)
order by 
	brand, segment;
	

select 
	brand,
	segment,
	sum(quantity)
from
	sales
group by
	brand,
	cube(segment)
order by 
	brand, segment;