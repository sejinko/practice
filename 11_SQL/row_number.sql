select 
	a.product_name,
	b.group_name,
	a.price,
	row_number () over (partition by b.group_name order by a.price)
from 
	product a
inner join product_group b 
on (a.group_id = b.group_id);


select 
	a.product_name,
	b.group_name,
	a.price,
	rank() over (partition by b.group_name order by a.price)
from 
	product a
inner join product_group b 
on (a.group_id = b.group_id);


select 
	a.product_name,
	b.group_name,
	a.price,
	dense_rank() over (partition by b.group_name order by a.price)
from 
	product a
inner join product_group b 
on (a.group_id = b.group_id);
	