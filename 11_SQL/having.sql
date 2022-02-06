select
	customer_id,
	sum(amount) as amount
from
	payment
group by customer_id;


select
	customer_id,
	sum(amount) as amount
from
	payment
group by customer_id
having sum(amount) > 200;


select 
	store_id,
	count(customer_id) as count
from
	customer
group by
	store_id;


select 
	store_id,
	count(customer_id) as count
from
	customer
group by
	store_id
having
	store_id = 1;


select 
	store_id,
	count(customer_id) as count
from
	customer
group by store_id 
having count(customer_id) > 300;


select 
	store_id,
	count(customer_id) as count
from
	customer
group by store_id 
having count > 300;
