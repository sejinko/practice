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


select 
	a.film_id,
	a.title,
	a.rental_rate
from
	film a,
	(select
		avg(rental_rate) as avg_rental_rate
	from film
	) b 
where a.rental_rate > b.avg_rengal_rate;
	