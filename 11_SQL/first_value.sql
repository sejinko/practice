select 
	a.product_name,
	b.group_name,
	a.price,
	first_value (a.price) over (partition by b.group_name order by a.price)
from 
	product a
inner join product_group b 
on (a.group_id = b.group_id);


select 
	a.product_name,
	b.group_name,
	a.price,
	last_value(a.price) over(partition by b.group_name order by a.price
	range between unbounded preceding and unbounded following)
	as highest_price_per_group
from product a
inner join product_group b 
on (a.group_id = b.group_id);