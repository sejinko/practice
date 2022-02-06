select 
	last_name,
	first_name
from
	customer
where 
	first_name = 'Jamie';
	

select 
	last_name,
	first_name
from
	customer
where 
	first_name = 'Jamie'
	and last_name = 'Rice';
	

select 
	customer_id,
	amount,
	payment_date
from
	payment
where 
	amount <= 1
	or amount >= 8;
	

select 
	film_id,
	title,
	release_year
from
	film
order by film_id
	limit 5;
	
select 
	film_id,
	title,
	release_year
from
	film
order by film_id
	limit 5;
	

select
	film_id,
	title,
	release_year
from
	film
order by film_id 
	limit 4
	offset 3;
	

select 
	film_id,
	title,
	rental_rate
from
	film
order by title desc
limit 10;


