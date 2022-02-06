select 
	brand,
	segment,
	sum(quantity)
from
	sales
group by
	rollup (brand, segment)
order by 
	brand, segment;
	

select 
	segment,
	brand,
	sum(quantity)
from
	sales
group by segment,
rollup (brand)
order by 
	segment, brand;