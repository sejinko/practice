-- ��� ���� ������ ���϶�
-- ��� �����ϴ� ������ �����ϰ��� �ϴ� ����

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

-- ����
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
		

-- a,b,c �ƹ����Գ� �ص� 3���� film_id�� ���� join�� �Ǵ� �ǰ�?
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


-- ��� sql ����
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