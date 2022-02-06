select 
	avg(rental_rate)
from
	film;
	

select 
	film_id,
	title,
	rental_rate
from
	film
where
	rental_rate > 2.98;
	

select 
	film_id,
	title,
	rental_rate
from
	film
where rental_rate >
(
	select 
		avg(rental_rate)
		from
			film
);


select 
	film_id,
	title,
	rental_rate
from 
	film
where rental_rate >
(
	select
		avg(rental_rate)
	from
		film
);
