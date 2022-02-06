select 
	title,
	length
from
	film
where 
	length >= any 
(
	select max(length)
	from film a,
		film_category b
	where a.film_id = b.film_id 
	group by b.category_id
		
)

