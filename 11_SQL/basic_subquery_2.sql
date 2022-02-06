select 
	film,
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
(
	select
		a.film_id,
		a.title,
		a.rental_rate,
		avg(a.rental_rate) over() as avg_rental_rate
	from
		film a
) a
where 
	a.rental_rate > a.avg_rental_rate;
		
	

select 
	a.film_id,
	a.title,
	a.rental_rate
from
(
	select
		a.film_id,
		a.title,
		a.rental_rate,
		avg(a.rental_rate) over() as avg_rental_rate
	from
		film a
) a
where 
	a.rental_rate > a.avg_rental_rate;
	




select 
	a.film_id,
	a.title,
	a.rental_rate
from
(
	select
		a.film_id,
		a.title,
		a.rental_rate,
		avg(a.rental_rate) over () as avg_rental_rate
	from
		film a
) a
where 
	a.rental_rate > avg_rental_rate;