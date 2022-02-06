-- 재고가 없는 집합을 구하라
-- 재고가 존재하는 집합을 제외하고자 하는 것입

select 
	film_id,
	title
from
	film
except
	select
		distinct inventory.film_id,
		title
	from
		inventory
	inner join
		film
	on film.film_id = inventory.film_id 
	order by title;
	


-- 정답 1
select 
	film_id,
	title
from 
	film a
where not exists
(
	select
		1
	from
		inventory b,
		film c
	where b.film_id = c.film_id
	and	a.film_id = c.film_id 
);


	
-- 더 좋은 정답
select 
	film_id,
	title
from 
	film a
where not exists
(
	select
		1
	from
		inventory b
	where 1=1
	and	a.film_id = b.film_id 
);


	