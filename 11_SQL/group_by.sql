select 
	customer_id
from
	payment;


select 
	customer_id
from
	payment
group by customer_id;


select 
	customer_id,
	sum(amount) as amount_sum
from
	payment
group by customer_id
order by sum(amount) desc;


select 
	customer_id,
	sum(amount) as amount_sum
from
	payment
group by customer_id
order by amount_sum desc;


select 
	customer_id,
	sum(amount) as amount_sum
from
	payment
group by customer_id 
order by 2 desc;


select 
	staff_id,
	count(payment_id) as count
from
	payment
group by staff_id;



 
