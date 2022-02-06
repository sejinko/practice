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

-- 정답
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
	and a.film_id = c.film_id 
);
		

-- a,b,c 아무렇게나 해도 3개의 film_id가 같이 join이 되는 건가?
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
		inventory c,
		film b
	where b.film_id = c.film_id
	and a.film_id = c.film_id 
);


-- 고급 sql 정답
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
	where 
		1=1
	and a.film_id = b.film_id
);