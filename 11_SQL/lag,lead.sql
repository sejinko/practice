
select 
	a.product_name,
	b.group_name,
	a.price,
	lag(a.price,1) over(partition by b.group_name order by a.price )as prev_price
from 
	product a
inner join product_group b 
on (a.group_id = b.group_id);



select 
	a.product_name,
	b.group_name,
	a.price,
	lag(a.price,1) over(partition by b.group_name order by a.price) as prev_price,
	a.price - lag(price, 1) over(partition by group_name order by a.price) as cur_prev_diff
from 
	product a
inner join product_group b 
on (a.group_id = b.group_id);



select 
	a.product_name,
	b.group_name,
	a.price,
	lead(a.price,1) over()
from
	product a
inner join product_group b 
on (a.group_id = b.group_id);



select 
	a.product_name,
	b.group_name,
	a.price,
	lead(a.price,1) over(partition by b.group_name order by a.price) next_price,
	a.price - lead(a.price,1) over(partition by b.group_name order by a.price) cur_next_diff
from
	product a
inner join product_group b 
on (a.group_id = b.group_id);