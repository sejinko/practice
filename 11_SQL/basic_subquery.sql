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
	)
order by
	film_id;
	

select
	avg(rental_rate)
from
	film;
	


-- °á°ú
select 
	a.film_id,
	a.title,
	a.rental_rate
from
(
	select 
		film_id,
		title,
		rental_rate,
		avg(rental_rate) over() as avg_rental_rate
	from 
		film
) a
where a.rental_rate > a.avg_rental_rate;


